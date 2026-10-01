from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mpc import run_mpc
from panda_trajopt.mujoco_sim import load_panda_simulation
from panda_trajopt.reaching import reaching_scene_geometry
from panda_trajopt.robustness import (
    RobustnessScenario,
    analyze_robustness,
    configure_simulation_mismatch,
    make_disturbance_schedule,
    make_target_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def test_robustness_schedules_switch_at_configured_steps() -> None:
    scenario = RobustnessScenario(
        name="test",
        disturbance_start_step=2,
        disturbance_duration_steps=2,
        disturbance_torque=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        goal_switch_step=3,
        goal_offset=(0.1, -0.1, 0.05),
    )
    scenario.validate(8)
    target = make_target_schedule(np.array([0.4, 0.0, 0.5]), scenario)
    disturbance = make_disturbance_schedule(scenario)

    assert np.allclose(target(2), [0.4, 0.0, 0.5])
    assert np.allclose(target(3), [0.5, -0.1, 0.55])
    assert np.allclose(disturbance(1), np.zeros(7))
    assert np.allclose(disturbance(2), np.arange(1.0, 8.0))
    assert np.allclose(disturbance(3), np.arange(1.0, 8.0))
    assert np.allclose(disturbance(4), np.zeros(7))


def test_mujoco_only_dynamics_mismatch_changes_plant() -> None:
    simulation = load_panda_simulation()
    hand_id = simulation.model.body("hand").id
    mass_before = simulation.model.body_mass[hand_id]
    friction_before = simulation.model.dof_frictionloss[simulation.arm_dof_ids].copy()
    damping_before = simulation.model.dof_damping[simulation.arm_dof_ids].copy()

    simulation.apply_dynamics_mismatch(
        payload_mass=0.5,
        joint_friction=0.2,
        damping_scale=1.4,
    )

    assert simulation.model.body_mass[hand_id] == pytest.approx(mass_before + 0.5)
    assert np.allclose(
        simulation.model.dof_frictionloss[simulation.arm_dof_ids],
        friction_before + 0.2,
    )
    assert np.allclose(
        simulation.model.dof_damping[simulation.arm_dof_ids],
        damping_before * 1.4,
    )


def test_mpc_records_dynamic_goal_and_external_disturbance() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    config = replace(
        config,
        reaching=replace(config.reaching, target_position=(0.53, 0.02, 0.55)),
        obstacle=replace(config.obstacle, center_position=(-0.4, 0.0, 0.5)),
        mpc=replace(config.mpc, horizon_steps=6, simulation_steps=5, max_iterations=2),
    )
    scenario = RobustnessScenario(
        name="integration",
        disturbance_start_step=1,
        disturbance_duration_steps=1,
        disturbance_torque=(0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        goal_switch_step=2,
        goal_offset=(0.0, 0.005, 0.0),
        payload_mass=0.1,
        joint_friction=0.01,
        damping_scale=1.05,
    )
    panda = load_panda_arm(config.robot)
    _, _, target, rotation, obstacle = reaching_scene_geometry(panda, config)
    simulation = load_panda_simulation(
        target_position=target,
        target_rotation=rotation,
        obstacle_position=obstacle,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )
    configure_simulation_mismatch(simulation, scenario)

    result = run_mpc(
        panda,
        config,
        simulation,
        target_position_schedule=make_target_schedule(target, scenario),
        external_torque_schedule=make_disturbance_schedule(scenario),
    )
    metrics = analyze_robustness(result, scenario, config.trajectory.time_step)

    result.validate()
    assert result.grasp_center_positions.shape == (config.mpc.simulation_steps + 1, 3)
    assert result.target_positions.shape == (config.mpc.simulation_steps + 1, 3)
    assert result.goal_errors.shape == (config.mpc.simulation_steps + 1,)
    assert np.allclose(result.external_torques[1], scenario.disturbance_torque)
    assert np.allclose(result.external_torques[0], np.zeros(7))
    assert np.allclose(result.target_positions[2], target)
    assert np.allclose(result.target_positions[3], target + np.asarray(scenario.goal_offset))
    assert metrics.disturbance_peak_error is not None
    assert metrics.goal_change_initial_error is not None
