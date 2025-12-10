from __future__ import annotations
from typing import Tuple, FrozenSet

from .config import Position
from .models import World


def neighbors(world: World, pos: Position) -> Tuple[Position, ...]:
    x, y = pos
    candidates = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))

    def valid(p: Position) -> bool:
        return 0 <= p[0] < world.width and 0 <= p[1] < world.height

    return tuple(filter(lambda p: valid(p), candidates))

def free_neighbors(world: World,
                   pos: Position,
                   blocked: FrozenSet[Position]) -> Tuple[Position, ...]:
    return tuple(filter(lambda p: p not in blocked, neighbors(world, pos)))



def herbivore_positions(world: World) -> FrozenSet[Position]:
    return frozenset(map(lambda h: h.pos, world.herbivores))


def carnivore_positions(world: World) -> FrozenSet[Position]:
    return frozenset(map(lambda c: c.pos, world.carnivores))


def occupied_positions(world: World) -> FrozenSet[Position]:
    return world.obstacles | world.plants | herbivore_positions(world) | carnivore_positions(world)


def empty_neighbors(world: World, pos: Position) -> Tuple[Position, ...]:
    occ = occupied_positions(world)
    return tuple(filter(lambda p: p not in occ, neighbors(world, pos)))