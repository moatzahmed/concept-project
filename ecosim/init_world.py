from __future__ import annotations
from typing import Tuple

from .config import (
    Position,
    OBSTACLE_DENSITY,
    INITIAL_PLANTS,
    INITIAL_HERBIVORES,
    INITIAL_CARNIVORES,
    HERB_SYMBOL,
    CARN_SYMBOL,
)
from .models import World, Animal
from .rng import RNG, rng_int
from .config import GRID_WIDTH, GRID_HEIGHT  # optional if you want defaults elsewhere


def all_positions(width: int, height: int) -> Tuple[Position, ...]:
    def row(y: int, x: int) -> Tuple[Position, ...]:
        return () if x == width else ((x, y),) + row(y, x + 1)

    def rows(y: int) -> Tuple[Position, ...]:
        return () if y == height else row(y, 0) + rows(y + 1)

    return rows(0)


def random_subset_positions(rng: RNG,
                            positions: Tuple[Position, ...],
                            count: int) -> Tuple[Tuple[Position, ...], RNG]:
    if count <= 0 or not positions:
        return (), rng
    if count >= len(positions):
        return positions, rng

    idx, rng2 = rng_int(rng, 0, len(positions) - 1)
    chosen = positions[idx]
    remaining = positions[:idx] + positions[idx + 1:]
    rest, rng3 = random_subset_positions(rng2, remaining, count - 1)
    return (chosen,) + rest, rng3


def initial_world(width: int, height: int, rng: RNG) -> Tuple[World, RNG]:
    all_cells = all_positions(width, height)

    obstacle_count = int(width * height * OBSTACLE_DENSITY)
    obstacle_positions, rng1 = random_subset_positions(rng, all_cells, obstacle_count)
    obstacles = frozenset(obstacle_positions)

    remaining_cells = tuple(filter(lambda pos: pos not in obstacles, all_cells))

    plant_positions, rng2 = random_subset_positions(rng1, remaining_cells, INITIAL_PLANTS)
    plants = frozenset(plant_positions)

    remaining_cells2 = tuple(filter(lambda pos: pos not in plants, remaining_cells))

    def make_animals(rng_in: RNG,
                     remaining: Tuple[Position, ...],
                     n: int,
                     start_energy: int,
                     symbol: str,
                     next_id: int
                     ) -> Tuple[Tuple[Animal, ...], Tuple[Position, ...], RNG, int]:
        if n == 0 or not remaining:
            return (), remaining, rng_in, next_id
        idx, rng_a = rng_int(rng_in, 0, len(remaining) - 1)
        pos = remaining[idx]
        rest_positions = remaining[:idx] + remaining[idx + 1:]
        animal = Animal(id=next_id, x=pos[0], y=pos[1], energy=start_energy, symbol=symbol)
        others, final_positions, rng_final, final_id = make_animals(
            rng_a, rest_positions, n - 1, start_energy, symbol, next_id + 1
        )
        return (animal,) + others, final_positions, rng_final, final_id

    herbivores, rem_after_herb, rng3, next_id_h = make_animals(
        rng2, remaining_cells2, INITIAL_HERBIVORES, 12, HERB_SYMBOL, 1
    )
    carnivores, rem_after_carn, rng4, next_id_c = make_animals(
        rng3, rem_after_herb, INITIAL_CARNIVORES, 14, CARN_SYMBOL, next_id_h
    )

    world = World(
        width=width,
        height=height,
        plants=plants,
        obstacles=obstacles,
        herbivores=herbivores,
        carnivores=carnivores,
        next_id=next_id_c
    )
    return world, rng4
