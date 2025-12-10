from __future__ import annotations
from typing import Tuple

# --------------------------- Configuration --------------------------- #

GRID_WIDTH = 20
GRID_HEIGHT = 20

INITIAL_HERBIVORES = 22
INITIAL_CARNIVORES = 6
INITIAL_PLANTS = 35

OBSTACLE_DENSITY = 0.07

ENERGY_GAIN_FROM_PLANT = 5
ENERGY_GAIN_FROM_PREY = 18
ENERGY_COST_PER_TURN = 1

HERB_REPRODUCTION_THRESHOLD = 14
HERB_REPRODUCTION_COST = 6
HERB_REPRODUCTION_PROBABILITY = 0.30

CARN_REPRODUCTION_THRESHOLD = 12
CARN_REPRODUCTION_COST = 5
CARN_REPRODUCTION_PROBABILITY = 0.40

PLANT_REGROWTH_PROBABILITY = 0.025

RANDOM_SEED = 42

EMPTY_SYMBOL = "."
HERB_SYMBOL = "H"
CARN_SYMBOL = "C"
PLANT_SYMBOL = "*"
OBSTACLE_SYMBOL = "#"
HORIZONTAL_BORDER_CHAR = "-"
VERTICAL_BORDER_CHAR = "|"

Position = Tuple[int, int]

# --------------------------- Colors & Symbols --------------------------- #

RESET = "\033[0m"
COLOR_HERB = "\033[97m"     # bright white
COLOR_CARN = "\033[91m"     # bright red
COLOR_PLANT = "\033[92m"    # bright green
COLOR_OBSTACLE = "\033[94m" # bright blue
