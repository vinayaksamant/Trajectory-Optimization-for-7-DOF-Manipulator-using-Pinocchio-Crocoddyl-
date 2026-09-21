from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mujoco_sim import load_panda_simulation
from panda_trajopt.playback import replay_reaching_solution
from panda_trajopt.reaching import solve_reaching_problem

ROOT = Path(__file__).resolve().parents[1]


def test_crocoddyl_solution_tracks_in_mujoco() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    solution = solve_reaching_problem(panda, config)
    simulation = load_panda_simulation(
        target_position=solution.target_position,
        target_rotation=solution.target_rotation,
    )

    result = replay_reaching_solution(panda, config, solution, simulation)

    result.validate()
    assert result.actual_states.shape == solution.states.shape
    assert result.applied_controls.shape == solution.controls.shape
    assert np.allclose(result.actual_states[0], solution.states[0])
    assert result.minimum_joint_margin > 0
    assert result.maximum_torque_ratio <= 1.0
    assert result.saturated_control_steps == 0
    assert result.final_grasp_center_orientation_error < 1e-2
