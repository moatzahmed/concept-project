from __future__ import annotations
from typing import Tuple

from .models import World, Animal
from .functional import fmap
from .config import ENERGY_COST_PER_TURN


def apply_energy_and_cleanup(world: World) -> Tuple[World, Tuple[str, ...]]:
    def dec_energy(animal: Animal) -> Animal:
        return Animal(animal.id, animal.x, animal.y, animal.energy - ENERGY_COST_PER_TURN, animal.symbol)

    dec_herb = fmap(dec_energy, world.herbivores)
    dec_carn = fmap(dec_energy, world.carnivores)

    alive_herb = tuple(filter(lambda h: h.energy > 0, dec_herb))
    alive_carn = tuple(filter(lambda c: c.energy > 0, dec_carn))


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