from ecosim.simulation import run_pure_simulation, display_steps

if __name__ == "__main__":
    steps_data = run_pure_simulation(steps=200)
    display_steps(0, steps_data)
