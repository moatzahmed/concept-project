from __future__ import annotations
from typing import Tuple

from .models import World
from .config import (
    HERB_SYMBOL,
    CARN_SYMBOL,
    PLANT_SYMBOL,
    OBSTACLE_SYMBOL,
    EMPTY_SYMBOL,
    HORIZONTAL_BORDER_CHAR,
    VERTICAL_BORDER_CHAR,
    COLOR_HERB,
    COLOR_CARN,
    COLOR_PLANT,
    COLOR_OBSTACLE,
    RESET,
)
from .config import Position


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


def symbol_at(world: World, pos: Position) -> str:
    x, y = pos
    if (x, y) in world.obstacles:
        return OBSTACLE_SYMBOL
    if (x, y) in world.plants:
        return PLANT_SYMBOL

    if any(map(lambda h: h.pos == (x, y), world.herbivores)):
        return HERB_SYMBOL

    if any(map(lambda c: c.pos == (x, y), world.carnivores)):
        return CARN_SYMBOL

    return EMPTY_SYMBOL


def row_symbols(world: World, y: int, x: int) -> Tuple[str, ...]:
    if x == world.width:
        return ()
    sym = symbol_at(world, (x, y))
    return (sym,) + row_symbols(world, y, x + 1)


def colored_row(world: World, y: int) -> str:
    syms = row_symbols(world, y, 0)

    def color_seq(seq: Tuple[str, ...]) -> Tuple[str, ...]:
        if not seq:
            return ()
        head, tail = seq[0], seq[1:]
        return (color_symbol(head),) + color_seq(tail)

    colored = color_seq(syms)
    row_str = " ".join(colored)
    return f"{VERTICAL_BORDER_CHAR}{row_str}{VERTICAL_BORDER_CHAR}"


def all_rows(world: World, y: int) -> Tuple[str, ...]:
    if y == world.height:
        return ()
    return (colored_row(world, y),) + all_rows(world, y + 1)


def render_world(world: World) -> str:
    content_width = world.width * 2 - 1
    border = HORIZONTAL_BORDER_CHAR * (content_width + 2)

    inner_rows = all_rows(world, 0)

    def prepend(s: str, seq: Tuple[str, ...]) -> Tuple[str, ...]:
        return (s,) + seq

    def append(seq: Tuple[str, ...], s: str) -> Tuple[str, ...]:
        return seq + (s,)

    lines = append(prepend(border, inner_rows), border)
    return "\n".join(lines)
