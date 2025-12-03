"""
2D Grid Ecosystem Simulation with Obstacles.

Herbivores eat plants, carnivores eat herbivores, plants regrow, and animals
lose energy over time. Obstacles block movement and placement.

Run: python main.py
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

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


# --------------------------- Data Structures --------------------------- #

@dataclass
class Animal:
    id: int
    x: int
    y: int
    energy: int

    @property
    def pos(self) -> Position:
        return (self.x, self.y)

    def step_energy_cost(self) -> None:
        self.energy -= ENERGY_COST_PER_TURN

    def is_alive(self) -> bool:
        return self.energy > 0


class Herbivore(Animal):
    symbol = HERB_SYMBOL


class Carnivore(Animal):
    symbol = CARN_SYMBOL


# --------------------------- World Class --------------------------- #

class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.plants: Set[Position] = set()
        self.obstacles: Set[Position] = set()
        self.herbivores: List[Herbivore] = []
        self.carnivores: List[Carnivore] = []
        self._next_id = 1

    # -------- Initialization -------- #

    def seed_world(self) -> None:
        self._seed_obstacles()
        self._seed_plants(INITIAL_PLANTS)
        self._seed_animals(Herbivore, INITIAL_HERBIVORES, start_energy=12)
        self._seed_animals(Carnivore, INITIAL_CARNIVORES, start_energy=14)

    def _seed_obstacles(self) -> None:
        count = int(self.width * self.height * OBSTACLE_DENSITY)
        for pos in self._random_empty_positions(count):
            self.obstacles.add(pos)

    def _seed_plants(self, count: int) -> None:
        for pos in self._random_empty_positions(count):
            self.plants.add(pos)

    def _seed_animals(self, cls, count: int, start_energy: int) -> None:
        for pos in self._random_empty_positions(count):
            self._add_animal(cls, pos, start_energy)

    def _add_animal(self, cls, pos: Position, energy: int) -> None:
        if cls is Herbivore:
            self.herbivores.append(Herbivore(self._next_id, pos[0], pos[1], energy))
        else:
            self.carnivores.append(Carnivore(self._next_id, pos[0], pos[1], energy))
        self._next_id += 1

    def _random_empty_positions(self, count: int, forbidden: Iterable[Position] = ()) -> List[Position]:
        taken = (
            self.obstacles
            | self.plants
            | {h.pos for h in self.herbivores}
            | {c.pos for c in self.carnivores}
            | set(forbidden)
        )
        empties = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in taken
        ]
        random.shuffle(empties)
        return empties[:count]

    # -------- Simulation -------- #

    def update_world(self, logs: List[str]) -> None:
        self.move_animals(logs)
        self.handle_eating(logs)
        self.apply_energy_and_cleanup(logs)
        self.handle_reproduction(logs)
        self.regrow_plants(logs)

    # -------- Movement -------- #

    def move_animals(self, logs: List[str]) -> None:
        for herb in random.sample(self.herbivores, len(self.herbivores)):
            target = self._choose_move_for_herbivore(herb)
            if target:
                herb.x, herb.y = target

        for carn in random.sample(self.carnivores, len(self.carnivores)):
            target = self._choose_move_for_carnivore(carn)
            if target:
                carn.x, carn.y = target

    def _choose_move_for_herbivore(self, herb: Herbivore) -> Optional[Position]:
        neighbors = self._free_neighbors(herb.pos, blocked=self.obstacles | {
            h.pos for h in self.herbivores if h.id != herb.id
        })

        plant_cells = [p for p in neighbors if p in self.plants]
        if plant_cells:
            return random.choice(plant_cells)

        safe_neighbors = [p for p in neighbors if p not in {c.pos for c in self.carnivores}]
        return random.choice(safe_neighbors) if safe_neighbors else (random.choice(neighbors) if neighbors else None)

    def _choose_move_for_carnivore(self, carn: Carnivore) -> Optional[Position]:
        neighbors = self._free_neighbors(carn.pos, blocked=self.obstacles | {
            c.pos for c in self.carnivores if c.id != carn.id
        })

        prey_cells = [p for p in neighbors if p in {h.pos for h in self.herbivores}]
        if prey_cells:
            return random.choice(prey_cells)

        return random.choice(neighbors) if neighbors else None

    # -------- Eating -------- #

    def handle_eating(self, logs: List[str]) -> None:
        for herb in list(self.herbivores):
            if herb.pos in self.plants:
                self.plants.remove(herb.pos)
                herb.energy += ENERGY_GAIN_FROM_PLANT
                logs.append(f"Herbivore #{herb.id} ate a plant at {herb.pos}.")

        for carn in list(self.carnivores):
            prey = self._herbivore_at(carn.pos)
            if prey:
                self.herbivores.remove(prey)
                carn.energy += ENERGY_GAIN_FROM_PREY
                logs.append(f"Carnivore #{carn.id} ate Herbivore #{prey.id} at {carn.pos}.")

    # -------- Energy & Cleanup -------- #

    def apply_energy_and_cleanup(self, logs: List[str]) -> None:
        for herb in self.herbivores:
            herb.step_energy_cost()
        for carn in self.carnivores:
            carn.step_energy_cost()

        before_h = len(self.herbivores)
        before_c = len(self.carnivores)

        self.herbivores = [h for h in self.herbivores if h.is_alive()]
        self.carnivores = [c for c in self.carnivores if c.is_alive()]

        if before_h - len(self.herbivores):
            logs.append(f"{before_h - len(self.herbivores)} herbivore(s) died from starvation.")
        if before_c - len(self.carnivores):
            logs.append(f"{before_c - len(self.carnivores)} carnivore(s) died from starvation.")

    # -------- Reproduction -------- #

    def handle_reproduction(self, logs: List[str]) -> None:
        new_herbivores = []
        for herb in self.herbivores:
            if herb.energy >= HERB_REPRODUCTION_THRESHOLD and random.random() < HERB_REPRODUCTION_PROBABILITY:
                pos = self._empty_neighbor(herb.pos)
                if pos:
                    herb.energy -= HERB_REPRODUCTION_COST
                    baby = Herbivore(self._next_id, pos[0], pos[1],
                                     energy=HERB_REPRODUCTION_THRESHOLD // 2)
                    self._next_id += 1
                    new_herbivores.append(baby)
                    logs.append(f"Herbivore #{herb.id} reproduced (#{baby.id}) at {pos}.")

        new_carnivores = []
        for carn in self.carnivores:
            if carn.energy >= CARN_REPRODUCTION_THRESHOLD and random.random() < CARN_REPRODUCTION_PROBABILITY:
                pos = self._empty_neighbor(carn.pos)
                if pos:
                    carn.energy -= CARN_REPRODUCTION_COST
                    baby = Carnivore(self._next_id, pos[0], pos[1],
                                     energy=CARN_REPRODUCTION_THRESHOLD // 2)
                    self._next_id += 1
                    new_carnivores.append(baby)
                    logs.append(f"Carnivore #{carn.id} reproduced (#{baby.id}) at {pos}.")

        self.herbivores.extend(new_herbivores)
        self.carnivores.extend(new_carnivores)

    # -------- Plants -------- #

    def regrow_plants(self, logs: List[str]) -> None:
        empties = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.obstacles
            and (x, y) not in self.plants
            and (x, y) not in {h.pos for h in self.herbivores}
            and (x, y) not in {c.pos for c in self.carnivores}
        ]

        new_plants = 0
        for cell in empties:
            if random.random() < PLANT_REGROWTH_PROBABILITY:
                self.plants.add(cell)
                new_plants += 1

        if new_plants:
            logs.append(f"{new_plants} new plant(s) sprouted.")

    # -------- Helpers -------- #

    def _neighbors(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [(nx, ny) for (nx, ny) in candidates if 0 <= nx < self.width and 0 <= ny < self.height]

    def _free_neighbors(self, pos: Position, blocked: Set[Position]) -> List[Position]:
        return [p for p in self._neighbors(pos) if p not in blocked]

    def _empty_neighbor(self, pos: Position) -> Optional[Position]:
        occupied = self.obstacles | self.plants | {h.pos for h in self.herbivores} | {c.pos for c in self.carnivores}
        candidates = [p for p in self._neighbors(pos) if p not in occupied]
        return random.choice(candidates) if candidates else None

    def _herbivore_at(self, pos: Position) -> Optional[Herbivore]:
        for herb in self.herbivores:
            if herb.pos == pos:
                return herb
        return None

    # -------- Rendering -------- #

    def render(self) -> str:
        grid = [[EMPTY_SYMBOL for _ in range(self.width)] for _ in range(self.height)]

        for x, y in self.obstacles:
            grid[y][x] = OBSTACLE_SYMBOL
        for x, y in self.plants:
            grid[y][x] = PLANT_SYMBOL
        for herb in self.herbivores:
            grid[herb.y][herb.x] = HERB_SYMBOL
        for carn in self.carnivores:
            grid[carn.y][carn.x] = CARN_SYMBOL

        content_width = self.width * 2 - 1
        border = HORIZONTAL_BORDER_CHAR * (content_width + 2)

        lines = [border]
        for row in grid:
            colored_row = [color_symbol(sym) for sym in row]
            row_str = " ".join(colored_row)
            lines.append(f"{VERTICAL_BORDER_CHAR}{row_str}{VERTICAL_BORDER_CHAR}")
        lines.append(border)
        return "\n".join(lines)


# --------------------------- Simulation Loop --------------------------- #

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
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    world = World(GRID_WIDTH, GRID_HEIGHT)
    world.seed_world()

    print("Starting 2D ecosystem simulation...")
    print_legend()
    time.sleep(1)

    for step in range(1, TOTAL_STEPS + 1):
        logs: List[str] = []
        clear_screen()
        print(f"Step {step}")
        print(world.render())
        print(
            f"Herbivores: {len(world.herbivores)} | "
            f"Carnivores: {len(world.carnivores)} | "
            f"Plants: {len(world.plants)} | "
            f"Obstacles: {len(world.obstacles)}"
        )
        print_legend()

        world.update_world(logs)

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
