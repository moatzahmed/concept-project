"""
2D Grid Ecosystem Simulation with Obstacles.

Herbivores eat plants, carnivores eat herbivores, plants regrow, and animals
lose energy over time. Obstacles block movement and placement.

Run: python main.py
"""

from __future__ import annotations

import os
import random
import time
from typing import List

from config import (
    STEP_DELAY_SECONDS,
    TOTAL_STEPS,
    RANDOM_SEED,
    EMPTY_SYMBOL,
    HERB_SYMBOL,
    CARN_SYMBOL,
    PLANT_SYMBOL,
    OBSTACLE_SYMBOL,
    color_symbol,
)
from world import World


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_legend() -> None:
    print(
        "Legend: "
        f"{EMPTY_SYMBOL}=empty  "
        f"{color_symbol(HERB_SYMBOL)}=herbivore  "
        f"{color_symbol(CARN_SYMBOL)}=carnivore  "
        f"{color_symbol(PLANT_SYMBOL)}=plant  "
        f"{color_symbol(OBSTACLE_SYMBOL)}=obstacle"
    )


def run_simulation() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # World dimensions come from config inside World.seed_world usage
    world = World(width=20, height=20)
    world.seed_world()

    print("Starting 2D ecosystem simulation...")
    print_legend()
    time.sleep(1)

    for step in range(1, TOTAL_STEPS + 1):
        logs: List[str] = []
        clear_screen()
        print(f"Step {step}")
        print(world.render())
        print(
            f"Herbivores: {len(world.herbivores)} | "
            f"Carnivores: {len(world.carnivores)} | "
            f"Plants: {len(world.plants)} | "
            f"Obstacles: {len(world.obstacles)}"
        )
        print_legend()

        world.update_world(logs)

        if logs:
            print("\nEvents:")
            for entry in logs:
                print(f"- {entry}")

        if not world.herbivores and not world.carnivores:
            print("\nAll animals have died out. Simulation ended early.")
            break

        time.sleep(STEP_DELAY_SECONDS)


if __name__ == "__main__":
    run_simulation()
