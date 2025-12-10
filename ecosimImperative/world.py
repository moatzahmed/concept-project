from __future__ import annotations

import random
from typing import Iterable, List, Optional, Set, Tuple

from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    INITIAL_HERBIVORES,
    INITIAL_CARNIVORES,
    INITIAL_PLANTS,
    OBSTACLE_DENSITY,
    ENERGY_GAIN_FROM_PLANT,
    ENERGY_GAIN_FROM_PREY,
    HERB_REPRODUCTION_THRESHOLD,
    HERB_REPRODUCTION_COST,
    HERB_REPRODUCTION_PROBABILITY,
    CARN_REPRODUCTION_THRESHOLD,
    CARN_REPRODUCTION_COST,
    CARN_REPRODUCTION_PROBABILITY,
    PLANT_REGROWTH_PROBABILITY,
    EMPTY_SYMBOL,
    PLANT_SYMBOL,
    OBSTACLE_SYMBOL,
    HORIZONTAL_BORDER_CHAR,
    VERTICAL_BORDER_CHAR,
    Position,
    color_symbol,
)
from entities import Herbivore, Carnivore


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

    def _random_empty_positions(
        self,
        count: int,
        forbidden: Iterable[Position] = (),
    ) -> List[Position]:
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
        neighbors = self._free_neighbors(
            herb.pos,
            blocked=self.obstacles
            | {h.pos for h in self.herbivores if h.id != herb.id},
        )

        plant_cells = [p for p in neighbors if p in self.plants]
        if plant_cells:
            return random.choice(plant_cells)

        safe_neighbors = [p for p in neighbors if p not in {c.pos for c in self.carnivores}]
        return (
            random.choice(safe_neighbors)
            if safe_neighbors
            else (random.choice(neighbors) if neighbors else None)
        )

    def _choose_move_for_carnivore(self, carn: Carnivore) -> Optional[Position]:
        neighbors = self._free_neighbors(
            carn.pos,
            blocked=self.obstacles
            | {c.pos for c in self.carnivores if c.id != carn.id},
        )

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
                logs.append(
                    f"Carnivore #{carn.id} ate Herbivore #{prey.id} at {carn.pos}."
                )

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
        new_herbivores: List[Herbivore] = []
        for herb in self.herbivores:
            if (
                herb.energy >= HERB_REPRODUCTION_THRESHOLD
                and random.random() < HERB_REPRODUCTION_PROBABILITY
            ):
                pos = self._empty_neighbor(herb.pos)
                if pos:
                    herb.energy -= HERB_REPRODUCTION_COST
                    baby = Herbivore(
                        self._next_id,
                        pos[0],
                        pos[1],
                        energy=HERB_REPRODUCTION_THRESHOLD // 2,
                    )
                    self._next_id += 1
                    new_herbivores.append(baby)
                    logs.append(f"Herbivore #{herb.id} reproduced (#{baby.id}) at {pos}.")

        new_carnivores: List[Carnivore] = []
        for carn in self.carnivores:
            if (
                carn.energy >= CARN_REPRODUCTION_THRESHOLD
                and random.random() < CARN_REPRODUCTION_PROBABILITY
            ):
                pos = self._empty_neighbor(carn.pos)
                if pos:
                    carn.energy -= CARN_REPRODUCTION_COST
                    baby = Carnivore(
                        self._next_id,
                        pos[0],
                        pos[1],
                        energy=CARN_REPRODUCTION_THRESHOLD // 2,
                    )
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
        return [
            (nx, ny)
            for (nx, ny) in candidates
            if 0 <= nx < self.width and 0 <= ny < self.height
        ]

    def _free_neighbors(self, pos: Position, blocked: Set[Position]) -> List[Position]:
        return [p for p in self._neighbors(pos) if p not in blocked]

    def _empty_neighbor(self, pos: Position) -> Optional[Position]:
        occupied = (
            self.obstacles
            | self.plants
            | {h.pos for h in self.herbivores}
            | {c.pos for c in self.carnivores}
        )
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
            grid[herb.y][herb.x] = Herbivore.symbol
        for carn in self.carnivores:
            grid[carn.y][carn.x] = Carnivore.symbol

        content_width = self.width * 2 - 1
        border = HORIZONTAL_BORDER_CHAR * (content_width + 2)

        lines = [border]
        for row in grid:
            colored_row = [color_symbol(sym) for sym in row]
            row_str = " ".join(colored_row)
            lines.append(f"{VERTICAL_BORDER_CHAR}{row_str}{VERTICAL_BORDER_CHAR}")
        lines.append(border)
        return "\n".join(lines)
