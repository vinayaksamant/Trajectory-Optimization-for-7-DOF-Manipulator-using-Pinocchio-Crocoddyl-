from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_trajopt.collision import minimum_arm_obstacle_distance
from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mujoco_sim import PANDA_HOME
from panda_trajopt.reaching import solve_reaching_problem

ROOT = Path(__file__).resolve().parents[1]


def test_crocoddyl_forward_dynamics_matches_pinocchio() -> None:
    import crocoddyl
    import pinocchio as pin

    config = load_config(ROOT / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    model = panda.model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    costs = crocoddyl.CostModelSum(state, actuation.nu)
    differential = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, costs)
    data = differential.createData()
    q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    v = np.zeros(7)
    u = np.zeros(7)

    differential.calc(data, np.concatenate((q, v)), u)
    expected = pin.aba(model, model.createData(), q, v, u)

    assert np.allclose(data.xout, expected, atol=1e-12)


def test_box_fddp_reaching_solution_is_valid() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    config = replace(
        config,
        reaching=replace(config.reaching, target_position=(0.4745, 0.08, 0.6011)),
        obstacle=replace(config.obstacle, center_position=(0.5025, 0.052, 0.5731)),
    )
    panda = load_panda_arm(config.robot)

    solution = solve_reaching_problem(panda, config)

    solution.validate()
    assert solution.states.shape == (config.trajectory.horizon_steps + 1, 14)
    assert solution.controls.shape == (config.trajectory.horizon_steps, 7)
    assert solution.feedback_gains.shape == (config.trajectory.horizon_steps, 7, 14)
    assert solution.planned_grasp_center_positions.shape == (
        config.trajectory.horizon_steps + 1,
        3,
    )
    assert solution.planned_arm_obstacle_distances.shape == (config.trajectory.horizon_steps + 1,)
    assert len(solution.solver_cost_trace) > 1
    assert len(solution.solver_stopping_trace) > 1
    assert np.allclose(solution.target_rotation, solution.initial_rotation)
    assert solution.final_orientation_error < 1e-3
    assert solution.minimum_arm_obstacle_clearance >= -1e-3
    assert solution.minimum_arm_obstacle_distance >= config.obstacle.safety_margin - 1e-3
    assert solution.closest_collision_geometry.startswith("panda_")

    direct_path_distances = [
        minimum_arm_obstacle_distance(
            panda,
            (1.0 - fraction) * PANDA_HOME + fraction * solution.states[-1, :7],
            solution.obstacle_center,
            solution.obstacle_radius,
        ).distance
        for fraction in np.linspace(0.0, 1.0, 21)
    ]
    assert min(direct_path_distances) < 0.0
