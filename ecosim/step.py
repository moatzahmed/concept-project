from __future__ import annotations
from .models import World, StepResult
from .rng import RNG
from .movement import move_animals
from .eating import handle_eating
from .energy import apply_energy_and_cleanup
from .reproduction import handle_reproduction
from .plants import regrow_plants


def step_world(world: World, rng: RNG) -> StepResult:
    world1, rng1, logs_move = move_animals(world, rng)
    world2, logs_eat = handle_eating(world1)
    world3, logs_energy = apply_energy_and_cleanup(world2)
    world4, rng2, logs_repro = handle_reproduction(world3, rng1)
    world5, rng3, logs_plants = regrow_plants(world4, rng2)

    logs = logs_move + logs_eat + logs_energy + logs_repro + logs_plants
    return StepResult(world=world5, logs=logs, rng=rng3)
