from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
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
    panda = load_panda_arm(config.robot)

    solution = solve_reaching_problem(panda, config)

    solution.validate()
    assert solution.states.shape == (config.trajectory.horizon_steps + 1, 14)
    assert solution.controls.shape == (config.trajectory.horizon_steps, 7)
    assert solution.feedback_gains.shape == (config.trajectory.horizon_steps, 7, 14)
    assert np.allclose(solution.target_rotation, solution.initial_rotation)
    assert solution.final_orientation_error < 1e-3
