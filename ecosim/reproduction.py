from __future__ import annotations
from typing import Tuple

from .models import World, Animal
from .rng import RNG, rng_float, rng_choice
from .neighbors import empty_neighbors
from .config import (
    HERB_REPRODUCTION_THRESHOLD,
    HERB_REPRODUCTION_COST,
    HERB_REPRODUCTION_PROBABILITY,
    CARN_REPRODUCTION_THRESHOLD,
    CARN_REPRODUCTION_COST,
    CARN_REPRODUCTION_PROBABILITY,
    HERB_SYMBOL,
)
from .config import Position  # for type hints if needed


def handle_reproduction(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
    logs: Tuple[str, ...] = ()
    next_id = world.next_id

    def reproduce_animals(animals: Tuple[Animal, ...],
                          threshold: int,
                          cost: int,
                          probability: float,
                          rng_in: RNG,
                          next_id_in: int) -> Tuple[Tuple[Animal, ...], Tuple[Animal, ...], RNG, int, Tuple[str, ...]]:
        def loop(idx: int,
                 acc_parents: Tuple[Animal, ...],
                 babies: Tuple[Animal, ...],
                 logs_local: Tuple[str, ...],
                 rng_local: RNG,
                 next_id_local: int):
            if idx == len(animals):
                return acc_parents, babies, rng_local, next_id_local, logs_local
            parent = animals[idx]
            if parent.energy < threshold:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_local, next_id_local)
            p, rng_p = rng_float(rng_local)
            if p >= probability:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_p, next_id_local)

            empties = empty_neighbors(world, parent.pos)
            if not empties:
                return loop(idx + 1, acc_parents + (parent,), babies, logs_local, rng_p, next_id_local)

            pos_baby, rng_b = rng_choice(rng_p, empties)
            new_energy_parent = parent.energy - cost
            new_parent = Animal(parent.id, parent.x, parent.y, new_energy_parent, parent.symbol)
            baby = Animal(next_id_local, pos_baby[0], pos_baby[1], threshold // 2, parent.symbol)
            new_logs = logs_local + (f"{'Herbivore' if parent.symbol == HERB_SYMBOL else 'Carnivore'} "
                                     f"#{parent.id} reproduced (#{baby.id}) at {pos_baby}.",)
            return loop(
                idx + 1,
                acc_parents + (new_parent,),
                babies + (baby,),
                new_logs,
                rng_b,
                next_id_local + 1
            )

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
