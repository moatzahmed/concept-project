"""
2D Grid Ecosystem Simulation with Obstacles - Functional Style.

Herbivores eat plants, carnivores eat herbivores, plants regrow, and animals
lose energy over time. Obstacles block movement and placement.

Run: python main_functional.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

# --------------------------- Configuration --------------------------- #

GRID_WIDTH = 20
GRID_HEIGHT = 20

# --- POPULATION SIZES --- #
INITIAL_HERBIVORES = 22
INITIAL_CARNIVORES = 6
INITIAL_PLANTS = 35

OBSTACLE_DENSITY = 0.07

# --- ENERGY SYSTEM --- #
ENERGY_GAIN_FROM_PLANT = 5
ENERGY_GAIN_FROM_PREY = 18
ENERGY_COST_PER_TURN = 1

# --- HERBIVORE REPRODUCTION --- #
HERB_REPRODUCTION_THRESHOLD = 14
HERB_REPRODUCTION_COST = 6
HERB_REPRODUCTION_PROBABILITY = 0.30

# --- CARNIVORE REPRODUCTION --- #
CARN_REPRODUCTION_THRESHOLD = 12
CARN_REPRODUCTION_COST = 5
CARN_REPRODUCTION_PROBABILITY = 0.40

# --- PLANTS --- #
PLANT_REGROWTH_PROBABILITY = 0.025

STEP_DELAY_SECONDS = 2
TOTAL_STEPS = 200
RANDOM_SEED = 42

# Symbols
EMPTY_SYMBOL = "."
HERB_SYMBOL = "H"
CARN_SYMBOL = "C"
PLANT_SYMBOL = "*"
OBSTACLE_SYMBOL = "#"
HORIZONTAL_BORDER_CHAR = "-"
VERTICAL_BORDER_CHAR = "|"

# ANSI colors
RESET = "\033[0m"
COLOR_HERB = "\033[97m"     # bright white
COLOR_CARN = "\033[91m"     # bright red
COLOR_PLANT = "\033[92m"    # bright green
COLOR_OBSTACLE = "\033[94m" # bright blue


def color_symbol(symbol: str) -> str:
    if symbol == HERB_SYMBOL:
        return f"{COLOR_HERB}{symbol}{RESET}"
    if symbol == CARN_SYMBOL:
        return f"{COLOR_CARN}{symbol}{RESET}"
    if symbol == PLANT_SYMBOL:
        return f"{COLOR_PLANT}{symbol}{RESET}"
    if symbol == OBSTACLE_SYMBOL:
        return f"{COLOR_OBSTACLE}{symbol}{RESET}"
    return symbol


Position = Tuple[int, int]

# --------------------------- RNG (pure style) --------------------------- #

# Simple linear congruential generator (LCG)
LCG_M = 2 ** 31 - 1
LCG_A = 1103515245
LCG_C = 12345

Seed = int


def rand(seed: Seed) -> Tuple[Seed, float]:
    """Return (new_seed, random_float_in_[0,1))."""
    new_seed = (LCG_A * seed + LCG_C) % LCG_M
    return new_seed, new_seed / LCG_M


def rand_bool(seed: Seed, p: float) -> Tuple[Seed, bool]:
    seed, r = rand(seed)
    return seed, r < p


def rand_choice(seed: Seed, seq: List) -> Tuple[Seed, Optional[object]]:
    if not seq:
        return seed, None
    seed, r = rand(seed)
    idx = int(r * len(seq))
    if idx == len(seq):
        idx = len(seq) - 1
    return seed, seq[idx]


def rand_shuffle(seed: Seed, seq: List) -> Tuple[Seed, List]:
    lst = list(seq)
    n = len(lst)
    for i in range(n - 1, 0, -1):
        seed, r = rand(seed)
        j = int(r * (i + 1))
        lst[i], lst[j] = lst[j], lst[i]
    return seed, lst


# --------------------------- Data Structures --------------------------- #

@dataclass(frozen=True)
class Animal:
    id: int
    x: int
    y: int
    energy: int

    @property
    def pos(self) -> Position:
        return (self.x, self.y)


@dataclass(frozen=True)
class Herbivore(Animal):
    symbol: str = HERB_SYMBOL


@dataclass(frozen=True)
class Carnivore(Animal):
    symbol: str = CARN_SYMBOL


@dataclass(frozen=True)
class World:
    width: int
    height: int
    plants: FrozenSet[Position]
    obstacles: FrozenSet[Position]
    herbivores: Tuple[Herbivore, ...]
    carnivores: Tuple[Carnivore, ...]
    next_id: int


# --------------------------- Helpers --------------------------- #

def all_positions(width: int, height: int) -> List[Position]:
    return [(x, y) for x in range(width) for y in range(height)]


def animal_positions(animals: Iterable[Animal]) -> Set[Position]:
    return {a.pos for a in animals}


def neighbors(pos: Position, width: int, height: int) -> List[Position]:
    x, y = pos
    candidates = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    return [(nx, ny) for (nx, ny) in candidates if 0 <= nx < width and 0 <= ny < height]


def free_neighbors(pos: Position, width: int, height: int, blocked: Set[Position]) -> List[Position]:
    return [p for p in neighbors(pos, width, height) if p not in blocked]


def random_empty_positions(
    world: World,
    seed: Seed,
    count: int,
    forbidden: Iterable[Position] = (),
) -> Tuple[Seed, List[Position]]:
    taken = (
        set(world.obstacles)
        | set(world.plants)
        | animal_positions(world.herbivores)
        | animal_positions(world.carnivores)
        | set(forbidden)
    )
    empties = [p for p in all_positions(world.width, world.height) if p not in taken]
    seed, shuffled = rand_shuffle(seed, empties)
    return seed, shuffled[:count]


# --------------------------- World Initialization --------------------------- #

def make_empty_world(width: int, height: int) -> World:
    return World(
        width=width,
        height=height,
        plants=frozenset(),
        obstacles=frozenset(),
        herbivores=tuple(),
        carnivores=tuple(),
        next_id=1,
    )


def seed_obstacles(world: World, seed: Seed) -> Tuple[Seed, World]:
    count = int(world.width * world.height * OBSTACLE_DENSITY)
    seed, positions = random_empty_positions(world, seed, count)
    new_obstacles = frozenset(set(world.obstacles) | set(positions))
    return seed, World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=new_obstacles,
        herbivores=world.herbivores,
        carnivores=world.carnivores,
        next_id=world.next_id,
    )


def seed_plants(world: World, seed: Seed, count: int) -> Tuple[Seed, World]:
    seed, positions = random_empty_positions(world, seed, count)
    new_plants = frozenset(set(world.plants) | set(positions))
    return seed, World(
        width=world.width,
        height=world.height,
        plants=new_plants,
        obstacles=world.obstacles,
        herbivores=world.herbivores,
        carnivores=world.carnivores,
        next_id=world.next_id,
    )


def seed_animals(
    world: World,
    seed: Seed,
    cls,
    count: int,
    start_energy: int,
) -> Tuple[Seed, World]:
    seed, positions = random_empty_positions(world, seed, count)
    new_herbivores = list(world.herbivores)
    new_carnivores = list(world.carnivores)
    next_id = world.next_id

    for x, y in positions:
        if cls is Herbivore:
            new_herbivores.append(Herbivore(next_id, x, y, start_energy))
        else:
            new_carnivores.append(Carnivore(next_id, x, y, start_energy))
        next_id += 1

    return seed, World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=tuple(new_herbivores),
        carnivores=tuple(new_carnivores),
        next_id=next_id,
    )


def seed_world(width: int, height: int, seed: Seed) -> Tuple[Seed, World]:
    base = make_empty_world(width, height)
    seed, w1 = seed_obstacles(base, seed)
    seed, w2 = seed_plants(w1, seed, INITIAL_PLANTS)
    seed, w3 = seed_animals(w2, seed, Herbivore, INITIAL_HERBIVORES, start_energy=12)
    seed, w4 = seed_animals(w3, seed, Carnivore, INITIAL_CARNIVORES, start_energy=14)
    return seed, w4


# --------------------------- Simulation Steps --------------------------- #

def choose_move_for_herbivore(world: World, herb: Herbivore, seed: Seed) -> Tuple[Seed, Optional[Position]]:
    herb_positions = animal_positions(world.herbivores)
    carn_positions = animal_positions(world.carnivores)
    blocked = set(world.obstacles) | (herb_positions - {herb.pos})
    neigh = free_neighbors(herb.pos, world.width, world.height, blocked)
    plant_cells = [p for p in neigh if p in world.plants]
    if plant_cells:
        seed, target = rand_choice(seed, plant_cells)
        return seed, target
    safe_neighbors = [p for p in neigh if p not in carn_positions]
    if safe_neighbors:
        seed, target = rand_choice(seed, safe_neighbors)
        return seed, target
    if neigh:
        seed, target = rand_choice(seed, neigh)
        return seed, target
    return seed, None


def choose_move_for_carnivore(world: World, carn: Carnivore, seed: Seed) -> Tuple[Seed, Optional[Position]]:
    carn_positions = animal_positions(world.carnivores)
    herb_positions = animal_positions(world.herbivores)
    blocked = set(world.obstacles) | (carn_positions - {carn.pos})
    neigh = free_neighbors(carn.pos, world.width, world.height, blocked)
    prey_cells = [p for p in neigh if p in herb_positions]
    if prey_cells:
        seed, target = rand_choice(seed, prey_cells)
        return seed, target
    if neigh:
        seed, target = rand_choice(seed, neigh)
        return seed, target
    return seed, None


def move_animals(world: World, seed: Seed) -> Tuple[Seed, World]:
    # Move herbivores
    seed, herb_order = rand_shuffle(seed, list(world.herbivores))
    moved_herbs: List[Herbivore] = []
    for herb in herb_order:
        seed, target = choose_move_for_herbivore(world, herb, seed)
        if target is not None:
            new_x, new_y = target
        else:
            new_x, new_y = herb.x, herb.y
        moved_herbs.append(Herbivore(herb.id, new_x, new_y, herb.energy))

    # Move carnivores (seeing the moved herbivores)
    temp_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=tuple(moved_herbs),
        carnivores=world.carnivores,
        next_id=world.next_id,
    )

    seed, carn_order = rand_shuffle(seed, list(world.carnivores))
    moved_carns: List[Carnivore] = []
    for carn in carn_order:
        seed, target = choose_move_for_carnivore(temp_world, carn, seed)
        if target is not None:
            new_x, new_y = target
        else:
            new_x, new_y = carn.x, carn.y
        moved_carns.append(Carnivore(carn.id, new_x, new_y, carn.energy))

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=tuple(moved_herbs),
        carnivores=tuple(moved_carns),
        next_id=world.next_id,
    )
    return seed, new_world


def handle_eating(world: World) -> Tuple[World, List[str]]:
    logs: List[str] = []

    # Herbivores eat plants
    new_plants = set(world.plants)
    updated_herbs: List[Herbivore] = []
    for herb in world.herbivores:
        energy = herb.energy
        if herb.pos in new_plants:
            new_plants.remove(herb.pos)
            energy += ENERGY_GAIN_FROM_PLANT
            logs.append(f"Herbivore #{herb.id} ate a plant at {herb.pos}.")
        updated_herbs.append(Herbivore(herb.id, herb.x, herb.y, energy))

    # Carnivores eat herbivores
    remaining_herbs: List[Herbivore] = []
    eaten_ids: Set[int] = set()
    herb_by_pos: List[Herbivore] = list(updated_herbs)

    updated_carns: List[Carnivore] = []
    for carn in world.carnivores:
        prey_index = next(
            (i for i, h in enumerate(herb_by_pos) if h.pos == carn.pos and h.id not in eaten_ids),
            None,
        )
        energy = carn.energy
        if prey_index is not None:
            prey = herb_by_pos[prey_index]
            eaten_ids.add(prey.id)
            energy += ENERGY_GAIN_FROM_PREY
            logs.append(f"Carnivore #{carn.id} ate Herbivore #{prey.id} at {carn.pos}.")
        updated_carns.append(Carnivore(carn.id, carn.x, carn.y, energy))

    remaining_herbs = [h for h in herb_by_pos if h.id not in eaten_ids]

    new_world = World(
        width=world.width,
        height=world.height,
        plants=frozenset(new_plants),
        obstacles=world.obstacles,
        herbivores=tuple(remaining_herbs),
        carnivores=tuple(updated_carns),
        next_id=world.next_id,
    )
    return new_world, logs


def apply_energy_and_cleanup(world: World) -> Tuple[World, List[str]]:
    logs: List[str] = []
    new_herbs: List[Herbivore] = []
    new_carns: List[Carnivore] = []

    dead_h = 0
    dead_c = 0

    for herb in world.herbivores:
        new_energy = herb.energy - ENERGY_COST_PER_TURN
        if new_energy > 0:
            new_herbs.append(Herbivore(herb.id, herb.x, herb.y, new_energy))
        else:
            dead_h += 1

    for carn in world.carnivores:
        new_energy = carn.energy - ENERGY_COST_PER_TURN
        if new_energy > 0:
            new_carns.append(Carnivore(carn.id, carn.x, carn.y, new_energy))
        else:
            dead_c += 1

    if dead_h:
        logs.append(f"{dead_h} herbivore(s) died from starvation.")
    if dead_c:
        logs.append(f"{dead_c} carnivore(s) died from starvation.")

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=tuple(new_herbs),
        carnivores=tuple(new_carns),
        next_id=world.next_id,
    )
    return new_world, logs


def random_empty_neighbor(world: World, pos: Position, occupied: Set[Position], seed: Seed) -> Tuple[Seed, Optional[Position]]:
    neigh = neighbors(pos, world.width, world.height)
    candidates = [p for p in neigh if p not in occupied]
    if not candidates:
        return seed, None
    seed, choice = rand_choice(seed, candidates)
    return seed, choice


def handle_reproduction(world: World, seed: Seed) -> Tuple[Seed, World, List[str]]:
    logs: List[str] = []
    occupied: Set[Position] = (
        set(world.obstacles)
        | set(world.plants)
        | animal_positions(world.herbivores)
        | animal_positions(world.carnivores)
    )

    new_herbs: List[Herbivore] = []
    new_carns: List[Carnivore] = []
    next_id = world.next_id

    # Herbivores
    for herb in world.herbivores:
        parent_energy = herb.energy
        babies: List[Herbivore] = []
        if parent_energy >= HERB_REPRODUCTION_THRESHOLD:
            seed, will_reproduce = rand_bool(seed, HERB_REPRODUCTION_PROBABILITY)
            if will_reproduce:
                seed, baby_pos = random_empty_neighbor(world, herb.pos, occupied, seed)
                if baby_pos is not None:
                    parent_energy -= HERB_REPRODUCTION_COST
                    baby = Herbivore(
                        next_id, baby_pos[0], baby_pos[1],
                        HERB_REPRODUCTION_THRESHOLD // 2
                    )
                    next_id += 1
                    occupied.add(baby_pos)
                    babies.append(baby)
                    logs.append(f"Herbivore #{herb.id} reproduced (#{baby.id}) at {baby_pos}.")
        new_herbs.append(Herbivore(herb.id, herb.x, herb.y, parent_energy))
        new_herbs.extend(babies)

    # Carnivores
    for carn in world.carnivores:
        parent_energy = carn.energy
        babies_c: List[Carnivore] = []
        if parent_energy >= CARN_REPRODUCTION_THRESHOLD:
            seed, will_reproduce = rand_bool(seed, CARN_REPRODUCTION_PROBABILITY)
            if will_reproduce:
                seed, baby_pos = random_empty_neighbor(world, carn.pos, occupied, seed)
                if baby_pos is not None:
                    parent_energy -= CARN_REPRODUCTION_COST
                    baby = Carnivore(
                        next_id, baby_pos[0], baby_pos[1],
                        CARN_REPRODUCTION_THRESHOLD // 2
                    )
                    next_id += 1
                    occupied.add(baby_pos)
                    babies_c.append(baby)
                    logs.append(f"Carnivore #{carn.id} reproduced (#{baby.id}) at {baby_pos}.")
        new_carns.append(Carnivore(carn.id, carn.x, carn.y, parent_energy))
        new_carns.extend(babies_c)

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=tuple(new_herbs),
        carnivores=tuple(new_carns),
        next_id=next_id,
    )
    return seed, new_world, logs


def regrow_plants(world: World, seed: Seed) -> Tuple[Seed, World, List[str]]:
    logs: List[str] = []
    new_plants = set(world.plants)
    new_plants_count = 0
    occupied_animals = animal_positions(world.herbivores) | animal_positions(world.carnivores)

    for x in range(world.width):
        for y in range(world.height):
            pos = (x, y)
            if pos in world.obstacles or pos in new_plants or pos in occupied_animals:
                continue
            seed, r = rand(seed)
            if r < PLANT_REGROWTH_PROBABILITY:
                new_plants.add(pos)
                new_plants_count += 1

    if new_plants_count:
        logs.append(f"{new_plants_count} new plant(s) sprouted.")

    new_world = World(
        width=world.width,
        height=world.height,
        plants=frozenset(new_plants),
        obstacles=world.obstacles,
        herbivores=world.herbivores,
        carnivores=world.carnivores,
        next_id=world.next_id,
    )
    return seed, new_world, logs


def step_world(world: World, seed: Seed) -> Tuple[Seed, World, List[str]]:
    logs_all: List[str] = []

    seed, moved_world = move_animals(world, seed)
    eaten_world, logs = handle_eating(moved_world)
    logs_all.extend(logs)

    energy_world, logs = apply_energy_and_cleanup(eaten_world)
    logs_all.extend(logs)

    seed, repro_world, logs = handle_reproduction(energy_world, seed)
    logs_all.extend(logs)

    seed, plants_world, logs = regrow_plants(repro_world, seed)
    logs_all.extend(logs)

    return seed, plants_world, logs_all


# --------------------------- Rendering & IO --------------------------- #

def render_world(world: World) -> str:
    grid = [[EMPTY_SYMBOL for _ in range(world.width)] for _ in range(world.height)]
    for x, y in world.obstacles:
        grid[y][x] = OBSTACLE_SYMBOL
    for x, y in world.plants:
        grid[y][x] = PLANT_SYMBOL
    for herb in world.herbivores:
        grid[herb.y][herb.x] = HERB_SYMBOL
    for carn in world.carnivores:
        grid[carn.y][carn.x] = CARN_SYMBOL

    content_width = world.width * 2 - 1
    border = HORIZONTAL_BORDER_CHAR * (content_width + 2)

    lines = [border]
    for row in grid:
        colored_row = [color_symbol(sym) for sym in row]
        row_str = " ".join(colored_row)
        lines.append(f"{VERTICAL_BORDER_CHAR}{row_str}{VERTICAL_BORDER_CHAR}")
    lines.append(border)
    return "\n".join(lines)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_legend() -> None:
    print(
        "Legend: "
        f"{EMPTY_SYMBOL}=empty  "
        f"{color_symbol(HERB_SYMBOL)}=herbivore  "
        f"{color_symbol(CARN_SYMBOL)}=carnivore  "
        f"{color_symbol(PLANT_SYMBOL)}=plant  "
        f"{color_symbol(OBSTACLE_SYMBOL)}=obstacle"
    )


def run_simulation() -> None:
    seed = RANDOM_SEED if RANDOM_SEED is not None else 1
    seed, world = seed_world(GRID_WIDTH, GRID_HEIGHT, seed)

    print("Starting 2D ecosystem simulation (functional core)...")
    print_legend()
    time.sleep(1.0)

    for step in range(1, TOTAL_STEPS + 1):
        clear_screen()
        print(f"Step {step}")
        print(render_world(world))
        print(
            f"Herbivores: {len(world.herbivores)} | "
            f"Carnivores: {len(world.carnivores)} | "
            f"Plants: {len(world.plants)} | "
            f"Obstacles: {len(world.obstacles)}"
        )
        print_legend()

        seed, world, logs = step_world(world, seed)

        if logs:
            print("\nEvents:")
            for entry in logs:
                print(f"- {entry}")

        if not world.herbivores and not world.carnivores:
            print("\nAll animals have died out. Simulation ended early.")
            break

        time.sleep(STEP_DELAY_SECONDS)


if __name__ == "__main__":
    run_simulation()
