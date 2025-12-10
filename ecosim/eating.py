from __future__ import annotations
from typing import Optional, Tuple

from .models import World, Animal
from .config import ENERGY_GAIN_FROM_PLANT, ENERGY_GAIN_FROM_PREY
from .config import Position  


def herbivore_at(world: World, pos: Position) -> Optional[Animal]:
    matches = tuple(filter(lambda h: h.pos == pos, world.herbivores))
    return matches[0] if matches else None


def handle_eating(world: World) -> Tuple[World, Tuple[str, ...]]:
    # Herbivores eat plants
    def process_herb(idx: int,
                     plants,
                     animals: Tuple[Animal, ...],
                     logs: Tuple[str, ...]):
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
                     logs: Tuple[str, ...]):
        if idx == len(world.carnivores):
            return remaining_herb, carn_acc, logs
        carn = world.carnivores[idx]
        prey_candidates = tuple(filter(lambda h: h.pos == carn.pos, remaining_herb))
        if prey_candidates:
            prey = prey_candidates[0]
            new_herb = tuple(filter(lambda h: h.id != prey.id, remaining_herb))
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
