from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mujoco_sim import load_panda_simulation
from panda_trajopt.playback import replay_reaching_solution
from panda_trajopt.reaching import solve_reaching_problem
from panda_trajopt.visualization import plot_reaching_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def test_crocoddyl_solution_tracks_in_mujoco(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    config = replace(
        config,
        reaching=replace(config.reaching, target_position=(0.4745, 0.08, 0.6011)),
        obstacle=replace(config.obstacle, center_position=(0.5025, 0.052, 0.5731)),
    )
    panda = load_panda_arm(config.robot)
    solution = solve_reaching_problem(panda, config)
    simulation = load_panda_simulation(
        target_position=solution.target_position,
        target_rotation=solution.target_rotation,
        obstacle_position=solution.obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )

    result = replay_reaching_solution(panda, config, solution, simulation)

    result.validate()
    assert result.actual_states.shape == solution.states.shape
    assert result.applied_controls.shape == solution.controls.shape
    assert result.actual_grasp_center_positions.shape == (
        config.trajectory.horizon_steps + 1,
        3,
    )
    assert result.actual_arm_obstacle_distances.shape == (config.trajectory.horizon_steps + 1,)
    assert np.allclose(result.actual_states[0], solution.states[0])
    assert result.minimum_joint_margin > 0
    assert result.maximum_torque_ratio <= 1.0
    assert result.saturated_control_steps == 0
    assert result.final_grasp_center_orientation_error < 1e-2
    assert result.minimum_arm_obstacle_clearance >= -1e-3

    plot_path = tmp_path / "diagnostics.png"
    plot_reaching_diagnostics(config, solution, result, show=False, save_path=plot_path)
    assert plot_path.stat().st_size > 0
