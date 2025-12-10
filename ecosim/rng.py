from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence

from .config import Position

@dataclass(frozen=True)
class RNG:
    seed: int

LCG_A = 1103515245
LCG_C = 12345
LCG_M = 2 ** 31 - 1


def rng_next(rng: RNG) -> Tuple[int, RNG]:
    new_seed = (LCG_A * rng.seed + LCG_C) & LCG_M
    return new_seed, RNG(new_seed)


def rng_float(rng: RNG) -> Tuple[float, RNG]:
    n, rng2 = rng_next(rng)
    return n / (LCG_M + 1), rng2


def rng_int(rng: RNG, low: int, high: int) -> Tuple[int, RNG]:
    n, rng2 = rng_next(rng)
    span = high - low + 1
    return low + (n % span), rng2


def rng_choice(rng: RNG, seq: Sequence[Position]) -> Tuple[Position, RNG]:
    from .rng import rng_int  # to avoid circular in type checkers
    idx, rng2 = rng_int(rng, 0, len(seq) - 1)
    return seq[idx], rng2
