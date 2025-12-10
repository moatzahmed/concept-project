from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from config import (
    ENERGY_COST_PER_TURN,
    HERB_SYMBOL,
    CARN_SYMBOL,
    Position,
)


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
