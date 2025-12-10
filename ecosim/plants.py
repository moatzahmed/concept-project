from __future__ import annotations
from typing import Tuple, FrozenSet

from .models import World
from .rng import RNG, rng_float
from .neighbors import occupied_positions
from .config import PLANT_REGROWTH_PROBABILITY, Position
from .init_world import *

def regrow_plants(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
    occ = occupied_positions(world)

    empties = tuple(
    filter(
        lambda p: p not in occ and p not in world.obstacles,
        all_positions(world.width, world.height)
    )
)

    def loop(idx: int,
             plants_acc: FrozenSet[Position],
             rng_in: RNG,
             grown: int):
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
