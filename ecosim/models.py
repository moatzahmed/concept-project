from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, FrozenSet, NamedTuple

from .config import Position

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
    rng: "RNG"  # forward reference to rng.RNG
