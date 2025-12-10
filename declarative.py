from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, FrozenSet, Sequence, Callable, NamedTuple, Optional

import os
# --------------------------- Configuration --------------------------- #

GRID_WIDTH = 20
GRID_HEIGHT = 20

INITIAL_HERBIVORES = 22
INITIAL_CARNIVORES = 6
INITIAL_PLANTS = 35

OBSTACLE_DENSITY = 0.07

ENERGY_GAIN_FROM_PLANT = 5
ENERGY_GAIN_FROM_PREY = 18
ENERGY_COST_PER_TURN = 1

HERB_REPRODUCTION_THRESHOLD = 14
HERB_REPRODUCTION_COST = 6
HERB_REPRODUCTION_PROBABILITY = 0.30

CARN_REPRODUCTION_THRESHOLD = 12
CARN_REPRODUCTION_COST = 5
CARN_REPRODUCTION_PROBABILITY = 0.40

PLANT_REGROWTH_PROBABILITY = 0.025

RANDOM_SEED = 42

EMPTY_SYMBOL = "."
HERB_SYMBOL = "H"
CARN_SYMBOL = "C"
PLANT_SYMBOL = "*"
OBSTACLE_SYMBOL = "#"
HORIZONTAL_BORDER_CHAR = "-"
VERTICAL_BORDER_CHAR = "|"
Position = Tuple[int, int]







# --------------------------- Colors & Symbols (same as original) --------------------------- #

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


# --------------------------- Pure Rendering --------------------------- #

def symbol_at(world: World, pos: Position) -> str:
    """Return the *uncolored* symbol at a given position."""
    x, y = pos
    if (x, y) in world.obstacles:
        return OBSTACLE_SYMBOL
    if (x, y) in world.plants:
        return PLANT_SYMBOL

    # any() is allowed; no explicit loops
    if any(h.pos == (x, y) for h in world.herbivores):
        return HERB_SYMBOL
    if any(c.pos == (x, y) for c in world.carnivores):
        return CARN_SYMBOL

    return EMPTY_SYMBOL


def row_symbols(world: World, y: int, x: int) -> Tuple[str, ...]:
    """Build tuple of symbols for row y, from column x to end, via recursion."""
    if x == world.width:
        return ()
    sym = symbol_at(world, (x, y))
    return (sym,) + row_symbols(world, y, x + 1)


def colored_row(world: World, y: int) -> str:
    """Return one fully colored row (with border chars)."""
    syms = row_symbols(world, y, 0)
    # map color_symbol over syms via recursion (no list comp)
    def color_seq(seq: Tuple[str, ...]) -> Tuple[str, ...]:
        if not seq:
            return ()
        head, tail = seq[0], seq[1:]
        return (color_symbol(head),) + color_seq(tail)

    colored = color_seq(syms)
    row_str = " ".join(colored)
    return f"{VERTICAL_BORDER_CHAR}{row_str}{VERTICAL_BORDER_CHAR}"


def all_rows(world: World, y: int) -> Tuple[str, ...]:
    """Recursively build all grid rows from y to bottom."""
    if y == world.height:
        return ()
    return (colored_row(world, y),) + all_rows(world, y + 1)


def render_world(world: World) -> str:
    """
    Pure rendering function:
    - takes a World
    - returns a string of the colored grid
    """
    content_width = world.width * 2 - 1
    border = HORIZONTAL_BORDER_CHAR * (content_width + 2)

    inner_rows = all_rows(world, 0)
    # lines = [border] + list(inner_rows) + [border]  (but without lists)
    def prepend(s: str, seq: Tuple[str, ...]) -> Tuple[str, ...]:
        return (s,) + seq

    def append(seq: Tuple[str, ...], s: str) -> Tuple[str, ...]:
        return seq + (s,)

    lines = append(prepend(border, inner_rows), border)
    return "\n".join(lines)


# --------------------------- Pure RNG --------------------------- #
# Simple linear congruential generator to keep randomness pure:
# RNG is just a value that we thread through all functions.

@dataclass(frozen=True)
class RNG:
    seed: int


LCG_A = 1103515245
LCG_C = 12345
LCG_M = 2 ** 31 - 1


def rng_next(rng: RNG) -> Tuple[int, RNG]:
    new_seed = (LCG_A * rng.seed + LCG_C) & LCG_M
    return new_seed, RNG(new_seed)


def rng_float(rng: RNG) -> Tuple[float, RNG]:
    n, rng2 = rng_next(rng)
    return n / (LCG_M + 1), rng2


def rng_int(rng: RNG, low: int, high: int) -> Tuple[int, RNG]:
    n, rng2 = rng_next(rng)
    span = high - low + 1
    return low + (n % span), rng2


def rng_choice(rng: RNG, seq: Sequence[Position]) -> Tuple[Position, RNG]:
    idx, rng2 = rng_int(rng, 0, len(seq) - 1)
    return seq[idx], rng2


# --------------------------- Data Structures --------------------------- #

@dataclass(frozen=True)
class Animal:
    id: int
    x: int
    y: int
    energy: int
    symbol: str

    @property
    def pos(self) -> Position:
        return (self.x, self.y)


@dataclass(frozen=True)
class World:
    width: int
    height: int
    plants: FrozenSet[Position]
    obstacles: FrozenSet[Position]
    herbivores: Tuple[Animal, ...]
    carnivores: Tuple[Animal, ...]
    next_id: int


class StepResult(NamedTuple):
    world: World
    logs: Tuple[str, ...]
    rng: RNG


# --------------------------- Functional Helpers --------------------------- #

def foldl(func, acc, seq):
    return acc if not seq else foldl(func, func(acc, seq[0]), seq[1:])


def fmap(func, seq):
    return () if not seq else (func(seq[0]),) + fmap(func, seq[1:])


def filter_f(func, seq):
    if not seq:
        return ()
    head, tail = seq[0], seq[1:]
    rest = filter_f(func, tail)
    return ((head,) + rest) if func(head) else rest


# --------------------------- World Initialization --------------------------- #

def all_positions(width: int, height: int) -> Tuple[Position, ...]:
    # recursion over y then x
    def row(y: int, x: int) -> Tuple[Position, ...]:
        return () if x == width else ((x, y),) + row(y, x + 1)

    def rows(y: int) -> Tuple[Position, ...]:
        return () if y == height else row(y, 0) + rows(y + 1)

    return rows(0)


def random_subset_positions(rng: RNG,
                            positions: Tuple[Position, ...],
                            count: int) -> Tuple[Tuple[Position, ...], RNG]:
    """Pick 'count' distinct positions without mutation / loops."""
    if count <= 0 or not positions:
        return (), rng
    if count >= len(positions):
        # "randomly" permuting would require more code; for the purpose
        # of pure style, we'll just take them as-is here.
        return positions, rng

    # Pick one at random, then recurse on the remaining.
    idx, rng2 = rng_int(rng, 0, len(positions) - 1)
    chosen = positions[idx]
    remaining = positions[:idx] + positions[idx + 1:]
    rest, rng3 = random_subset_positions(rng2, remaining, count - 1)
    return (chosen,) + rest, rng3


def initial_world(width: int, height: int, rng: RNG) -> Tuple[World, RNG]:
    all_cells = all_positions(width, height)

    obstacle_count = int(width * height * OBSTACLE_DENSITY)
    obstacle_positions, rng1 = random_subset_positions(rng, all_cells, obstacle_count)
    obstacles = frozenset(obstacle_positions)

    remaining_cells = tuple(pos for pos in all_cells if pos not in obstacles)

    plant_positions, rng2 = random_subset_positions(rng1, remaining_cells, INITIAL_PLANTS)
    plants = frozenset(plant_positions)

    remaining_cells2 = tuple(pos for pos in remaining_cells if pos not in plants)

    def make_animals(rng_in: RNG,
                     remaining: Tuple[Position, ...],
                     n: int,
                     start_energy: int,
                     symbol: str,
                     next_id: int
                     ) -> Tuple[Tuple[Animal, ...], Tuple[Position, ...], RNG, int]:
        if n == 0 or not remaining:
            return (), remaining, rng_in, next_id
        idx, rng_a = rng_int(rng_in, 0, len(remaining) - 1)
        pos = remaining[idx]
        rest_positions = remaining[:idx] + remaining[idx + 1:]
        animal = Animal(id=next_id, x=pos[0], y=pos[1], energy=start_energy, symbol=symbol)
        others, final_positions, rng_final, final_id = make_animals(
            rng_a, rest_positions, n - 1, start_energy, symbol, next_id + 1
        )
        return (animal,) + others, final_positions, rng_final, final_id

    herbivores, rem_after_herb, rng3, next_id_h = make_animals(
        rng2, remaining_cells2, INITIAL_HERBIVORES, 12, HERB_SYMBOL, 1
    )
    carnivores, rem_after_carn, rng4, next_id_c = make_animals(
        rng3, rem_after_herb, INITIAL_CARNIVORES, 14, CARN_SYMBOL, next_id_h
    )

    world = World(
        width=width,
        height=height,
        plants=plants,
        obstacles=obstacles,
        herbivores=herbivores,
        carnivores=carnivores,
        next_id=next_id_c
    )
    return world, rng4


# --------------------------- Neighborhood Helpers --------------------------- #

def neighbors(world: World, pos: Position) -> Tuple[Position, ...]:
    x, y = pos
    candidates = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
    def valid(p: Position) -> bool:
        return 0 <= p[0] < world.width and 0 <= p[1] < world.height
    return tuple(p for p in candidates if valid(p))


def free_neighbors(world: World,
                   pos: Position,
                   blocked: FrozenSet[Position]) -> Tuple[Position, ...]:
    return tuple(p for p in neighbors(world, pos) if p not in blocked)


def herbivore_positions(world: World) -> FrozenSet[Position]:
    return frozenset(h.pos for h in world.herbivores)


def carnivore_positions(world: World) -> FrozenSet[Position]:
    return frozenset(c.pos for c in world.carnivores)


def occupied_positions(world: World) -> FrozenSet[Position]:
    return world.obstacles | world.plants | herbivore_positions(world) | carnivore_positions(world)


# --------------------------- Movement --------------------------- #

def choose_move_for_herbivore(world: World,
                              herb: Animal,
                              rng: RNG) -> Tuple[Position, RNG]:
    blocked = world.obstacles | frozenset(
        h.pos for h in world.herbivores if h.id != herb.id
    )
    neigh = free_neighbors(world, herb.pos, blocked)
    if not neigh:
        return herb.pos, rng

    plant_cells = tuple(p for p in neigh if p in world.plants)
    if plant_cells:
        chosen, rng2 = rng_choice(rng, plant_cells)
        return chosen, rng2

    carn_positions = carnivore_positions(world)
    safe_neighbors = tuple(p for p in neigh if p not in carn_positions)
    if safe_neighbors:
        chosen, rng2 = rng_choice(rng, safe_neighbors)
        return chosen, rng2

    # fallback: any neighbor
    chosen, rng2 = rng_choice(rng, neigh)
    return chosen, rng2


def choose_move_for_carnivore(world: World,
                              carn: Animal,
                              rng: RNG) -> Tuple[Position, RNG]:
    blocked = world.obstacles | frozenset(
        c.pos for c in world.carnivores if c.id != carn.id
    )
    neigh = free_neighbors(world, carn.pos, blocked)
    if not neigh:
        return carn.pos, rng

    herb_positions = herbivore_positions(world)
    prey_cells = tuple(p for p in neigh if p in herb_positions)
    if prey_cells:
        chosen, rng2 = rng_choice(rng, prey_cells)
        return chosen, rng2

    chosen, rng2 = rng_choice(rng, neigh)
    return chosen, rng2


def move_animals(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
    # In this pure version, all moves are based on the *current* world
    # snapshot (synchronous update); this is slightly different from
    # the original in-place sequential update.

    def move_herb_seq(animals: Tuple[Animal, ...],
                      idx: int,
                      acc: Tuple[Animal, ...],
                      rng_in: RNG) -> Tuple[Tuple[Animal, ...], RNG]:
        if idx == len(animals):
            return acc, rng_in
        herb = animals[idx]
        new_pos, rng2 = choose_move_for_herbivore(world, herb, rng_in)
        new_herb = Animal(herb.id, new_pos[0], new_pos[1], herb.energy, herb.symbol)
        return move_herb_seq(animals, idx + 1, acc + (new_herb,), rng2)

    def move_carn_seq(animals: Tuple[Animal, ...],
                      idx: int,
                      acc: Tuple[Animal, ...],
                      rng_in: RNG) -> Tuple[Tuple[Animal, ...], RNG]:
        if idx == len(animals):
            return acc, rng_in
        carn = animals[idx]
        new_pos, rng2 = choose_move_for_carnivore(world, carn, rng_in)
        new_carn = Animal(carn.id, new_pos[0], new_pos[1], carn.energy, carn.symbol)
        return move_carn_seq(animals, idx + 1, acc + (new_carn,), rng2)

    moved_herbivores, rng1 = move_herb_seq(world.herbivores, 0, (), rng)
    moved_carnivores, rng2 = move_carn_seq(world.carnivores, 0, (), rng1)

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=moved_herbivores,
        carnivores=moved_carnivores,
        next_id=world.next_id
    )
    return new_world, rng2, ()


# --------------------------- Eating --------------------------- #

def herbivore_at(world: World, pos: Position) -> Optional[Animal]:
    candidates = tuple(h for h in world.herbivores if h.pos == pos)
    return candidates[0] if candidates else None


def handle_eating(world: World) -> Tuple[World, Tuple[str, ...]]:
    # Herbivores eat plants
    def process_herb(idx: int,
                     plants: FrozenSet[Position],
                     animals: Tuple[Animal, ...],
                     logs: Tuple[str, ...]) -> Tuple[FrozenSet[Position], Tuple[Animal, ...], Tuple[str, ...]]:
        if idx == len(world.herbivores):
            return plants, animals, logs
        herb = world.herbivores[idx]
        if herb.pos in plants:
            new_plants = plants - {herb.pos}
            new_herb = Animal(herb.id, herb.x, herb.y, herb.energy + ENERGY_GAIN_FROM_PLANT, herb.symbol)
            new_logs = logs + (f"Herbivore #{herb.id} ate a plant at {herb.pos}.",)
            return process_herb(idx + 1, new_plants, animals + (new_herb,), new_logs)
        else:
            return process_herb(idx + 1, plants, animals + (herb,), logs)

    plants_after_herb, herbivores_after_herb, logs1 = process_herb(0, world.plants, (), ())

    # Carnivores eat herbivores
    def process_carn(idx: int,
                     remaining_herb: Tuple[Animal, ...],
                     carn_acc: Tuple[Animal, ...],
                     logs: Tuple[str, ...]) -> Tuple[Tuple[Animal, ...], Tuple[Animal, ...], Tuple[str, ...]]:
        if idx == len(world.carnivores):
            return remaining_herb, carn_acc, logs
        carn = world.carnivores[idx]
        prey_candidates = tuple(h for h in remaining_herb if h.pos == carn.pos)
        if prey_candidates:
            prey = prey_candidates[0]
            new_herb = tuple(h for h in remaining_herb if h.id != prey.id)
            new_carn = Animal(carn.id, carn.x, carn.y, carn.energy + ENERGY_GAIN_FROM_PREY, carn.symbol)
            new_logs = logs + (f"Carnivore #{carn.id} ate Herbivore #{prey.id} at {carn.pos}.",)
            return process_carn(idx + 1, new_herb, carn_acc + (new_carn,), new_logs)
        else:
            return process_carn(idx + 1, remaining_herb, carn_acc + (carn,), logs)

    herbivores_after_carn, carnivores_after_carn, logs2 = process_carn(0, herbivores_after_herb, (), ())

    new_world = World(
        width=world.width,
        height=world.height,
        plants=plants_after_herb,
        obstacles=world.obstacles,
        herbivores=herbivores_after_carn,
        carnivores=carnivores_after_carn,
        next_id=world.next_id
    )
    return new_world, logs1 + logs2


# --------------------------- Energy & Cleanup --------------------------- #

def apply_energy_and_cleanup(world: World) -> Tuple[World, Tuple[str, ...]]:
    def dec_energy(animal: Animal) -> Animal:
        return Animal(animal.id, animal.x, animal.y, animal.energy - ENERGY_COST_PER_TURN, animal.symbol)

    dec_herb = fmap(dec_energy, world.herbivores)
    dec_carn = fmap(dec_energy, world.carnivores)

    alive_herb = tuple(h for h in dec_herb if h.energy > 0)
    alive_carn = tuple(c for c in dec_carn if c.energy > 0)

    dead_h = len(dec_herb) - len(alive_herb)
    dead_c = len(dec_carn) - len(alive_carn)

    logs: Tuple[str, ...] = ()
    if dead_h > 0:
        logs = logs + (f"{dead_h} herbivore(s) died from starvation.",)
    if dead_c > 0:
        logs = logs + (f"{dead_c} carnivore(s) died from starvation.",)

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=alive_herb,
        carnivores=alive_carn,
        next_id=world.next_id
    )
    return new_world, logs


# --------------------------- Reproduction --------------------------- #

def empty_neighbors(world: World, pos: Position) -> Tuple[Position, ...]:
    occ = occupied_positions(world)
    return tuple(p for p in neighbors(world, pos) if p not in occ)


def handle_reproduction(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
    logs: Tuple[str, ...] = ()
    next_id = world.next_id

    def reproduce_animals(animals: Tuple[Animal, ...],
                          threshold: int,
                          cost: int,
                          probability: float,
                          rng_in: RNG,
                          next_id_in: int) -> Tuple[Tuple[Animal, ...], Tuple[Animal, ...], RNG, int, Tuple[str, ...]]:
        # returns (updated_parents, babies, rng_out, next_id_out, logs)
        def loop(idx: int,
                 acc_parents: Tuple[Animal, ...],
                 babies: Tuple[Animal, ...],
                 logs_local: Tuple[str, ...],
                 rng_local: RNG,
                 next_id_local: int
                 ) -> Tuple[Tuple[Animal, ...], Tuple[Animal, ...], RNG, int, Tuple[str, ...]]:
            if idx == len(animals):
                return acc_parents, babies, rng_local, next_id_local, logs_local
            parent = animals[idx]
            if parent.energy < threshold:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_local, next_id_local)
            p, rng_p = rng_float(rng_local)
            if p >= probability:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_p, next_id_local)

            # attempt reproduction
            empties = empty_neighbors(world, parent.pos)
            if not empties:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_p, next_id_local)

            pos_baby, rng_b = rng_choice(rng_p, empties)
            new_energy_parent = parent.energy - cost
            new_parent = Animal(parent.id, parent.x, parent.y, new_energy_parent, parent.symbol)
            baby = Animal(next_id_local, pos_baby[0], pos_baby[1], threshold // 2, parent.symbol)
            new_logs = logs_local + (f"{'Herbivore' if parent.symbol == HERB_SYMBOL else 'Carnivore'} "
                                     f"#{parent.id} reproduced (#{baby.id}) at {pos_baby}.",)
            return loop(idx + 1,
                        acc_parents + (new_parent,),
                        babies + (baby,),
                        new_logs,
                        rng_b,
                        next_id_local + 1)

        return loop(0, (), (), (), rng_in, next_id_in)

    herb_parents, herb_babies, rng1, next_id1, logs_h = reproduce_animals(
        world.herbivores,
        HERB_REPRODUCTION_THRESHOLD,
        HERB_REPRODUCTION_COST,
        HERB_REPRODUCTION_PROBABILITY,
        rng,
        next_id
    )

    carn_parents, carn_babies, rng2, next_id2, logs_c = reproduce_animals(
        world.carnivores,
        CARN_REPRODUCTION_THRESHOLD,
        CARN_REPRODUCTION_COST,
        CARN_REPRODUCTION_PROBABILITY,
        rng1,
        next_id1
    )

    new_world = World(
        width=world.width,
        height=world.height,
        plants=world.plants,
        obstacles=world.obstacles,
        herbivores=herb_parents + herb_babies,
        carnivores=carn_parents + carn_babies,
        next_id=next_id2
    )

    return new_world, rng2, logs_h + logs_c


# --------------------------- Plants Regrowth --------------------------- #

def regrow_plants(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
    occ = occupied_positions(world)

    empties = tuple(
        (x, y)
        for x in range(world.width)
        for y in range(world.height)
        if (x, y) not in occ and (x, y) not in world.obstacles
    )

    def loop(idx: int,
             plants_acc: FrozenSet[Position],
             rng_in: RNG,
             grown: int) -> Tuple[FrozenSet[Position], RNG, int]:
        if idx == len(empties):
            return plants_acc, rng_in, grown
        cell = empties[idx]
        p, rng2 = rng_float(rng_in)
        if p < PLANT_REGROWTH_PROBABILITY:
            return loop(idx + 1, plants_acc | {cell}, rng2, grown + 1)
        else:
            return loop(idx + 1, plants_acc, rng2, grown)

    new_plants, rng2, count_grown = loop(0, world.plants, rng, 0)
    logs: Tuple[str, ...] = ()
    if count_grown:
        logs = (f"{count_grown} new plant(s) sprouted.",)

    new_world = World(
        width=world.width,
        height=world.height,
        plants=new_plants,
        obstacles=world.obstacles,
        herbivores=world.herbivores,
        carnivores=world.carnivores,
        next_id=world.next_id
    )
    return new_world, rng2, logs


# --------------------------- One Pure Step --------------------------- #

def step_world(world: World, rng: RNG) -> StepResult:
    # movement
    world1, rng1, logs_move = move_animals(world, rng)

    # eating
    world2, logs_eat = handle_eating(world1)

    # energy + deaths
    world3, logs_energy = apply_energy_and_cleanup(world2)

    # reproduction
    world4, rng2, logs_repro = handle_reproduction(world3, rng1)

    # plants regrow
    world5, rng3, logs_plants = regrow_plants(world4, rng2)

    logs = logs_move + logs_eat + logs_energy + logs_repro + logs_plants
    
    return StepResult(world=world5, logs=logs, rng=rng3)


# --------------------------- Pure Simulation Runner --------------------------- #

def run_pure_simulation(steps: int,
                        width: int = GRID_WIDTH,
                        height: int = GRID_HEIGHT,
                        seed: int = RANDOM_SEED
                        ) -> Tuple[Tuple[World, Tuple[str, ...]], ...]:
    """
    Fully pure: returns a tuple of (world, logs) for each step.
    No I/O, no global mutation, no loops.
    """
    rng0 = RNG(seed)
    world0, rng1 = initial_world(width, height, rng0)

    def loop(step: int,
             world: World,
             rng_in: RNG,
             acc: Tuple[Tuple[World, Tuple[str, ...]], ...]
             ) -> Tuple[Tuple[World, Tuple[str, ...]], ...]:
        if step == 0 or (not world.herbivores and not world.carnivores):
            return acc
        result = step_world(world, rng_in)
        new_pair = (result.world, result.logs)
        return loop(step - 1, result.world, result.rng, acc + (new_pair,))

    return loop(steps, world0, rng1, ())




if __name__ == "__main__":
    from time import sleep

    steps_data = run_pure_simulation(steps=200)

    def display_steps(idx: int, data: Tuple[Tuple[World, Tuple[str, ...]], ...]) -> None:
        if idx == len(data):
            return

        world, logs = data[idx]
        os.system("cls" if os.name == "nt" else "clear")

        print(f"Step {idx + 1}")
        print(render_world(world))
        print(
            f"Herbivores: {len(world.herbivores)} | "
            f"Carnivores: {len(world.carnivores)} | "
            f"Plants: {len(world.plants)} | "
            f"Obstacles: {len(world.obstacles)}"
        )
        if logs:
            print("\nEvents:")
            for line in logs:   # <- You *can* also eliminate this if needed
                print("-", line)

        sleep(1)
        display_steps(idx + 1, data)

    display_steps(0, steps_data)

