"""First bounded Crocoddyl problem: move the Panda hand to a nearby target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_trajopt.config import ProjectConfig
from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PANDA_HOME


@dataclass(frozen=True)
class ReachingSolution:
    states: np.ndarray
    controls: np.ndarray
    feedback_gains: np.ndarray
    initial_position: np.ndarray
    initial_rotation: np.ndarray
    target_position: np.ndarray
    target_rotation: np.ndarray
    final_position: np.ndarray
    final_rotation: np.ndarray
    converged: bool
    iterations: int
    cost: float
    stopping_value: float
    final_position_error: float
    final_orientation_error: float
    final_speed: float
    minimum_joint_margin: float
    maximum_torque_ratio: float

    def validate(self) -> None:
        failures: list[str] = []
        if not self.converged:
            failures.append("BoxFDDP did not report convergence")
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.controls)):
            failures.append("trajectory contains a non-finite value")
        if self.final_position_error > 1e-3:
            failures.append(f"final grasp-center error is {self.final_position_error:.3e} m")
        if self.final_orientation_error > 1e-3:
            failures.append(
                f"final grasp-center orientation error is {self.final_orientation_error:.3e} rad"
            )
        if self.final_speed > 1e-2:
            failures.append(f"final joint speed norm is {self.final_speed:.3e} rad/s")
        if self.minimum_joint_margin < -1e-9:
            failures.append(f"joint limit exceeded by {-self.minimum_joint_margin:.3e} rad")
        if self.maximum_torque_ratio > 1.0 + 1e-9:
            failures.append(f"torque limit ratio is {self.maximum_torque_ratio:.6f}")
        if failures:
            raise ValueError("Invalid reaching solution: " + "; ".join(failures))


def rotation_distance(rotation: np.ndarray, reference: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    cosine = (np.trace(reference.T @ rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _state_cost(
    crocoddyl: object,
    state: object,
    state_reference: np.ndarray,
    control_dimension: int,
    terminal: bool,
) -> object:
    residual = crocoddyl.ResidualModelState(state, state_reference, control_dimension)
    position_weight = 0.01 if terminal else 0.1
    velocity_weight = 10.0 if terminal else 1.0
    weights = np.concatenate((np.full(7, position_weight), np.full(7, velocity_weight)))
    activation = crocoddyl.ActivationModelWeightedQuad(weights)
    return crocoddyl.CostModelResidual(state, activation, residual)


def _joint_limit_cost(
    crocoddyl: object,
    state: object,
    model: object,
    state_reference: np.ndarray,
    control_dimension: int,
) -> object:
    residual = crocoddyl.ResidualModelState(state, state_reference, control_dimension)
    lower = np.concatenate((model.lowerPositionLimit - state_reference[:7], np.full(7, -1e6)))
    upper = np.concatenate((model.upperPositionLimit - state_reference[:7], np.full(7, 1e6)))
    activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lower, upper))
    return crocoddyl.CostModelResidual(state, activation, residual)


def solve_reaching_problem(
    panda: PandaModel,
    config: ProjectConfig,
) -> ReachingSolution:
    """Solve a two-second, torque-limited hand-translation problem."""
    import crocoddyl
    import pinocchio as pin

    model = panda.model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    if state.nx != 14 or actuation.nu != 7:
        raise ValueError("Expected state dimension 14 and control dimension 7")

    initial_state = np.concatenate((PANDA_HOME, np.zeros(7)))
    pin_data = model.createData()
    pin.framesForwardKinematics(model, pin_data, PANDA_HOME)
    initial_placement = pin_data.oMf[panda.end_effector_frame_id]
    initial_position = initial_placement.translation.copy()
    initial_rotation = initial_placement.rotation.copy()
    target_position = initial_position + np.asarray(config.reaching.target_offset)
    target_rotation = initial_rotation @ pin.rpy.rpyToMatrix(
        np.asarray(config.reaching.target_orientation_rpy)
    )

    def make_costs(terminal: bool) -> object:
        costs = crocoddyl.CostModelSum(state, actuation.nu)
        frame_residual = crocoddyl.ResidualModelFrameTranslation(
            state,
            panda.end_effector_frame_id,
            target_position,
            actuation.nu,
        )
        frame_cost = crocoddyl.CostModelResidual(state, frame_residual)
        frame_weight = (
            config.costs.terminal_end_effector_position
            if terminal
            else config.costs.end_effector_position
        )
        costs.addCost("gripper_center_translation", frame_cost, frame_weight)

        rotation_residual = crocoddyl.ResidualModelFrameRotation(
            state,
            panda.end_effector_frame_id,
            target_rotation,
            actuation.nu,
        )
        rotation_weight = (
            config.costs.terminal_end_effector_orientation
            if terminal
            else config.costs.end_effector_orientation
        )
        costs.addCost(
            "gripper_center_orientation",
            crocoddyl.CostModelResidual(state, rotation_residual),
            rotation_weight,
        )

        state_weight = (
            config.costs.terminal_state_regularization
            if terminal
            else config.costs.state_regularization
        )
        costs.addCost(
            "state_regularization",
            _state_cost(crocoddyl, state, initial_state, actuation.nu, terminal),
            state_weight,
        )
        costs.addCost(
            "joint_limits",
            _joint_limit_cost(crocoddyl, state, model, initial_state, actuation.nu),
            config.costs.joint_limits,
        )
        if not terminal:
            control_residual = crocoddyl.ResidualModelControl(state, actuation.nu)
            costs.addCost(
                "control_regularization",
                crocoddyl.CostModelResidual(state, control_residual),
                config.costs.control_regularization,
            )
        return costs

    running_differential = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, make_costs(terminal=False)
    )
    running_model = crocoddyl.IntegratedActionModelEuler(
        running_differential, config.trajectory.time_step
    )
    terminal_differential = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, make_costs(terminal=True)
    )
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_differential, 0.0)

    torque_limits = np.asarray(config.robot.torque_limits)
    running_model.u_lb = -torque_limits
    running_model.u_ub = torque_limits

    problem = crocoddyl.ShootingProblem(
        initial_state,
        [running_model] * config.trajectory.horizon_steps,
        terminal_model,
    )
    initial_states = [initial_state.copy() for _ in range(problem.T + 1)]
    initial_controls = problem.quasiStatic(initial_states[:-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    converged = solver.solve(
        initial_states,
        initial_controls,
        config.reaching.max_iterations,
        False,
        config.reaching.stopping_threshold,
    )

    states = np.asarray(solver.xs).copy()
    controls = np.asarray(solver.us).copy()
    feedback_gains = np.asarray(solver.K).copy()
    final_configuration = states[-1, :7]
    pin.framesForwardKinematics(model, pin_data, final_configuration)
    final_position = pin_data.oMf[panda.end_effector_frame_id].translation.copy()
    final_rotation = pin_data.oMf[panda.end_effector_frame_id].rotation.copy()

    lower_margins = states[:, :7] - model.lowerPositionLimit
    upper_margins = model.upperPositionLimit - states[:, :7]
    maximum_torque_ratio = float(np.max(np.abs(controls) / torque_limits))
    solution = ReachingSolution(
        states=states,
        controls=controls,
        feedback_gains=feedback_gains,
        initial_position=initial_position,
        initial_rotation=initial_rotation,
        target_position=target_position,
        target_rotation=target_rotation,
        final_position=final_position,
        final_rotation=final_rotation,
        converged=bool(converged),
        iterations=int(solver.iter),
        cost=float(solver.cost),
        stopping_value=float(solver.stop),
        final_position_error=float(np.linalg.norm(final_position - target_position)),
        final_orientation_error=rotation_distance(final_rotation, target_rotation),
        final_speed=float(np.linalg.norm(states[-1, 7:])),
        minimum_joint_margin=float(min(np.min(lower_margins), np.min(upper_margins))),
        maximum_torque_ratio=maximum_torque_ratio,
    )
    solution.validate()
    return solution
