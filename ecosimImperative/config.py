from typing import Tuple

# --------------------------- Configuration --------------------------- #

GRID_WIDTH = 20
GRID_HEIGHT = 20

# --- POPULATION SIZES --- #
INITIAL_HERBIVORES = 22
INITIAL_CARNIVORES = 6
INITIAL_PLANTS = 35

OBSTACLE_DENSITY = 0.07

# --- ENERGY SYSTEM --- #
ENERGY_GAIN_FROM_PLANT = 5
ENERGY_GAIN_FROM_PREY = 18
ENERGY_COST_PER_TURN = 1

# --- HERBIVORE REPRODUCTION --- #
HERB_REPRODUCTION_THRESHOLD = 14
HERB_REPRODUCTION_COST = 6
HERB_REPRODUCTION_PROBABILITY = 0.30

# --- CARNIVORE REPRODUCTION --- #
CARN_REPRODUCTION_THRESHOLD = 12
CARN_REPRODUCTION_COST = 5
CARN_REPRODUCTION_PROBABILITY = 0.40

# --- PLANTS --- #
PLANT_REGROWTH_PROBABILITY = 0.025

STEP_DELAY_SECONDS = 0.2
TOTAL_STEPS = 200
RANDOM_SEED = 42

# Symbols
EMPTY_SYMBOL = "."
HERB_SYMBOL = "H"
CARN_SYMBOL = "C"
PLANT_SYMBOL = "*"
OBSTACLE_SYMBOL = "#"
HORIZONTAL_BORDER_CHAR = "-"
VERTICAL_BORDER_CHAR = "|"

# ANSI colors
RESET = "\033[0m"
COLOR_HERB = "\033[97m"     # bright white
COLOR_CARN = "\033[91m"     # bright red
COLOR_PLANT = "\033[92m"    # bright green
COLOR_OBSTACLE = "\033[94m" # bright blue

Position = Tuple[int, int]


def color_symbol(symbol: str) -> str:
    if symbol == HERB_SYMBOL:
        return f"{COLOR_HERB}{symbol}{RESET}"
    if symbol == CARN_SYMBOL:
        return f"{COLOR_CARN}{symbol}{RESET}"
    if symbol == PLANT_SYMBOL:
        return f"{COLOR_PLANT}{symbol}{RESET}"
    if symbol == OBSTACLE_SYMBOL:
        return f"{COLOR_OBSTACLE}{symbol}{RESET}"
    return symbol