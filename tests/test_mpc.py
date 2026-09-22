from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mpc import run_mpc
from panda_trajopt.mujoco_sim import load_panda_simulation
from panda_trajopt.reaching import reaching_scene_geometry

ROOT = Path(__file__).resolve().parents[1]


def test_box_fddp_mpc_replans_from_mujoco_state() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    config = replace(
        config,
        reaching=replace(config.reaching, target_position=(0.53, 0.02, 0.55)),
        obstacle=replace(config.obstacle, center_position=(-0.4, 0.0, 0.5)),
        mpc=replace(config.mpc, horizon_steps=6, simulation_steps=3, max_iterations=2),
    )
    panda = load_panda_arm(config.robot)
    _, _, target_position, target_rotation, obstacle_center = reaching_scene_geometry(panda, config)
    simulation = load_panda_simulation(
        target_position=target_position,
        target_rotation=target_rotation,
        obstacle_position=obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )

    replanned_steps: list[int] = []
    simulation_callback_count = 0

    def record_replan(step: int, _solution: object) -> None:
        replanned_steps.append(step)

    def record_simulation_step(_simulation: object) -> None:
        nonlocal simulation_callback_count
        simulation_callback_count += 1

    result = run_mpc(
        panda,
        config,
        simulation,
        on_replan=record_replan,
        after_simulation_step=record_simulation_step,
    )

    result.validate()
    assert result.solver_name == "box_fddp"
    assert result.states.shape == (config.mpc.simulation_steps + 1, 14)
    assert result.controls.shape == (config.mpc.simulation_steps, 7)
    assert result.replan_times.shape == (config.mpc.simulation_steps,)
    assert result.solver_iterations.shape == (config.mpc.simulation_steps,)
    assert result.solver_converged.shape == (config.mpc.simulation_steps,)
    assert np.all(result.replan_times > 0.0)
    assert result.final_goal_error < result.initial_goal_error
    assert result.minimum_arm_obstacle_clearance >= -1e-3
    assert result.maximum_torque_ratio <= 1.0
    assert result.saturated_control_steps == 0
    assert replanned_steps == list(range(config.mpc.simulation_steps))
    expected_substeps = round(config.trajectory.time_step / simulation.model.opt.timestep)
    assert simulation_callback_count == config.mpc.simulation_steps * expected_substeps


def test_ilqr_style_solver_can_drive_mpc() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    config = replace(
        config,
        reaching=replace(config.reaching, target_position=(0.53, 0.02, 0.55)),
        obstacle=replace(config.obstacle, center_position=(-0.4, 0.0, 0.5)),
        mpc=replace(config.mpc, horizon_steps=6, simulation_steps=2, max_iterations=2),
    )
    panda = load_panda_arm(config.robot)
    _, _, target_position, target_rotation, obstacle_center = reaching_scene_geometry(panda, config)
    simulation = load_panda_simulation(
        target_position=target_position,
        target_rotation=target_rotation,
        obstacle_position=obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )

    result = run_mpc(panda, config, simulation, solver_kind="ilqr")

    result.validate()
    assert result.solver_name == "ilqr"
    assert result.final_goal_error < result.initial_goal_error
