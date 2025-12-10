from __future__ import annotations
from typing import Tuple
import os
from time import sleep

from .config import GRID_WIDTH, GRID_HEIGHT, RANDOM_SEED
from .models import World
from .rng import RNG
from .init_world import initial_world
from .step import step_world
from .render import render_world


def run_pure_simulation(steps: int,
                        width: int = GRID_WIDTH,
                        height: int = GRID_HEIGHT,
                        seed: int = RANDOM_SEED
                        ) -> Tuple[Tuple[World, Tuple[str, ...]], ...]:
    rng0 = RNG(seed)
    world0, rng1 = initial_world(width, height, rng0)

    def loop(step: int,
             world: World,
             rng_in: RNG,
             acc: Tuple[Tuple[World, Tuple[str, ...]], ...]):
        if step == 0 or (not world.herbivores and not world.carnivores):
            return acc
        result = step_world(world, rng_in)
        new_pair = (result.world, result.logs)
        return loop(step - 1, result.world, result.rng, acc + (new_pair,))

    return loop(steps, world0, rng1, ())


def display_steps(idx: int, data: Tuple[Tuple[World, Tuple[str, ...]], ...]) -> None:
    if idx == len(data):
        return

    world, logs = data[idx]
    os.system("cls" if os.name == "nt" else "clear")

    print(f"Step {idx + 1}")
    print(render_world(world))
    print(
        f"Herbivores: {len(world.herbivores)} | "
        f"Carnivores: {len(world.carnivores)} | "
        f"Plants: {len(world.plants)} | "
        f"Obstacles: {len(world.obstacles)}"
    )
    if logs:
        print("\nEvents:")
        for line in logs:
            print("-", line)

    sleep(1)
    display_steps(idx + 1, data)


if __name__ == "__main__":
    steps_data = run_pure_simulation(steps=200)
    display_steps(0, steps_data)
