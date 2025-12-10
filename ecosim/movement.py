from __future__ import annotations
from typing import Tuple

from .config import Position
from .models import World, Animal
from .neighbors import (
    free_neighbors,
    carnivore_positions,
    herbivore_positions,
)
from .rng import RNG, rng_choice


def choose_move_for_herbivore(world: World,
                              herb: Animal,
                              rng: RNG) -> Tuple[Position, RNG]:
    # blocked = world.obstacles | frozenset(
    #     h.pos for h in world.herbivores if h.id != herb.id
    # )
    blocked_other_herbivores = frozenset(
        map(
            lambda h: h.pos,
            filter(lambda h: h.id != herb.id, world.herbivores)
        )
    )
    blocked = world.obstacles | blocked_other_herbivores

    neigh = free_neighbors(world, herb.pos, blocked)
    if not neigh:
        return herb.pos, rng

    # plant_cells = tuple(p for p in neigh if p in world.plants)
    plant_cells = tuple(filter(lambda p: p in world.plants, neigh))
    if plant_cells:
        chosen, rng2 = rng_choice(rng, plant_cells)
        return chosen, rng2

    carn_positions = carnivore_positions(world)

    # safe_neighbors = tuple(p for p in neigh if p not in carn_positions)
    safe_neighbors = tuple(filter(lambda p: p not in carn_positions, neigh))
    if safe_neighbors:
        chosen, rng2 = rng_choice(rng, safe_neighbors)
        return chosen, rng2

    chosen, rng2 = rng_choice(rng, neigh)
    return chosen, rng2


def choose_move_for_carnivore(world: World,
                              carn: Animal,
                              rng: RNG) -> Tuple[Position, RNG]:
    # blocked = world.obstacles | frozenset(
    #     c.pos for c in world.carnivores if c.id != carn.id
    # )
    blocked_other_carnivores = frozenset(
        map(
            lambda c: c.pos,
            filter(lambda c: c.id != carn.id, world.carnivores)
        )
    )
    blocked = world.obstacles | blocked_other_carnivores

    neigh = free_neighbors(world, carn.pos, blocked)
    if not neigh:
        return carn.pos, rng

    herb_positions_set = herbivore_positions(world)

    # prey_cells = tuple(p for p in neigh if p in herb_positions_set)
    prey_cells = tuple(filter(lambda p: p in herb_positions_set, neigh))
    if prey_cells:
        chosen, rng2 = rng_choice(rng, prey_cells)
        return chosen, rng2

    chosen, rng2 = rng_choice(rng, neigh)
    return chosen, rng2



def move_animals(world: World, rng: RNG) -> Tuple[World, RNG, Tuple[str, ...]]:
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
