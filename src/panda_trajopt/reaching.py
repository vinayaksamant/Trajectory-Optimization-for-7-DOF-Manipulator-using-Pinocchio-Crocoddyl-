"""Bounded Crocoddyl pose reaching with gripper-center obstacle avoidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_trajopt.collision import (
    make_full_arm_obstacle_residual,
    minimum_arm_obstacle_distance,
)
from panda_trajopt.config import ProjectConfig
from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PANDA_HOME


@dataclass(frozen=True)
class ReachingSolution:
    states: np.ndarray
    controls: np.ndarray
    feedback_gains: np.ndarray
    planned_grasp_center_positions: np.ndarray
    planned_arm_obstacle_distances: np.ndarray
    solver_cost_trace: np.ndarray
    solver_stopping_trace: np.ndarray
    initial_position: np.ndarray
    initial_rotation: np.ndarray
    target_position: np.ndarray
    target_rotation: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: float
    obstacle_safety_margin: float
    final_position: np.ndarray
    final_rotation: np.ndarray
    converged: bool
    iterations: int
    cost: float
    stopping_value: float
    final_position_error: float
    final_orientation_error: float
    minimum_arm_obstacle_distance: float
    minimum_arm_obstacle_clearance: float
    closest_collision_geometry: str
    final_speed: float
    minimum_joint_margin: float
    maximum_torque_ratio: float

    def validate(self) -> None:
        failures: list[str] = []
        if not self.converged:
            failures.append(
                "BoxFDDP did not report convergence "
                f"after {self.iterations} iterations (stop={self.stopping_value:.3e})"
            )
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.controls)):
            failures.append("trajectory contains a non-finite value")
        if self.planned_grasp_center_positions.shape != (self.states.shape[0], 3):
            failures.append("planned grasp-center path has unexpected dimensions")
        if self.planned_arm_obstacle_distances.shape != (self.states.shape[0],):
            failures.append("planned obstacle-distance trace has unexpected dimensions")
        if self.solver_cost_trace.size == 0 or self.solver_stopping_trace.size == 0:
            failures.append("solver convergence history is empty")
        if self.final_position_error > 1e-3:
            failures.append(f"final grasp-center error is {self.final_position_error:.3e} m")
        if self.final_orientation_error > 1e-3:
            failures.append(
                f"final grasp-center orientation error is {self.final_orientation_error:.3e} rad"
            )
        if self.minimum_arm_obstacle_clearance < -1e-3:
            failures.append(
                "Panda collision geometry enters the obstacle safety margin by "
                f"{-self.minimum_arm_obstacle_clearance:.3e} m"
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


def reaching_scene_geometry(
    panda: PandaModel, config: ProjectConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return initial pose, target pose, and obstacle center in the world frame."""
    import pinocchio as pin

    data = panda.model.createData()
    pin.framesForwardKinematics(panda.model, data, PANDA_HOME)
    initial_placement = data.oMf[panda.end_effector_frame_id]
    initial_position = initial_placement.translation.copy()
    initial_rotation = initial_placement.rotation.copy()
    target_position = np.asarray(config.reaching.target_position)
    target_rotation = initial_rotation @ pin.rpy.rpyToMatrix(
        np.asarray(config.reaching.target_orientation_rpy)
    )
    obstacle_center = np.asarray(config.obstacle.center_position)
    return (
        initial_position,
        initial_rotation,
        target_position,
        target_rotation,
        obstacle_center,
    )


def solve_reaching_problem(
    panda: PandaModel,
    config: ProjectConfig,
    *,
    validate_solution: bool = True,
) -> ReachingSolution:
    """Solve a torque-limited gripper pose task around a spherical obstacle."""
    import crocoddyl
    import pinocchio as pin

    model = panda.model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    if state.nx != 14 or actuation.nu != 7:
        raise ValueError("Expected state dimension 14 and control dimension 7")

    initial_state = np.concatenate((PANDA_HOME, np.zeros(7)))
    pin_data = model.createData()
    (
        initial_position,
        initial_rotation,
        target_position,
        target_rotation,
        obstacle_center,
    ) = reaching_scene_geometry(panda, config)
    initial_obstacle_distance = minimum_arm_obstacle_distance(
        panda, PANDA_HOME, obstacle_center, config.obstacle.radius
    )
    if initial_obstacle_distance.distance <= config.obstacle.safety_margin:
        raise ValueError(
            "The initial full-arm pose is inside the obstacle safety margin: "
            f"{initial_obstacle_distance.geometry_name} has "
            f"{initial_obstacle_distance.distance:.3e} m clearance"
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

        if not terminal:
            obstacle_residual = make_full_arm_obstacle_residual(
                crocoddyl,
                state,
                panda,
                obstacle_center,
                config.obstacle.radius,
                config.obstacle.activation_distance,
                actuation.nu,
            )
            obstacle_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(
                    np.array([config.obstacle.activation_distance]),
                    np.array([np.inf]),
                )
            )
            costs.addCost(
                "full_arm_obstacle",
                crocoddyl.CostModelResidual(state, obstacle_activation, obstacle_residual),
                config.costs.obstacle_avoidance,
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
    solver.th_stop = config.reaching.stopping_threshold
    solver_logger = crocoddyl.CallbackLogger()
    solver.setCallbacks([solver_logger])
    converged = solver.solve(
        initial_states,
        initial_controls,
        config.reaching.max_iterations,
        False,
    )

    states = np.asarray(solver.xs).copy()
    controls = np.asarray(solver.us).copy()
    feedback_gains = np.asarray(solver.K).copy()
    final_configuration = states[-1, :7]
    pin.framesForwardKinematics(model, pin_data, final_configuration)
    final_position = pin_data.oMf[panda.end_effector_frame_id].translation.copy()
    final_rotation = pin_data.oMf[panda.end_effector_frame_id].rotation.copy()

    planned_positions: list[np.ndarray] = []
    planned_distances: list[float] = []
    minimum_distance = np.inf
    closest_geometry = ""
    for configuration in states[:, :7]:
        pin.framesForwardKinematics(model, pin_data, configuration)
        planned_positions.append(pin_data.oMf[panda.end_effector_frame_id].translation.copy())
        report = minimum_arm_obstacle_distance(
            panda,
            configuration,
            obstacle_center,
            config.obstacle.radius,
        )
        planned_distances.append(report.distance)
        if report.distance < minimum_distance:
            minimum_distance = report.distance
            closest_geometry = report.geometry_name

    lower_margins = states[:, :7] - model.lowerPositionLimit
    upper_margins = model.upperPositionLimit - states[:, :7]
    maximum_torque_ratio = float(np.max(np.abs(controls) / torque_limits))
    solution = ReachingSolution(
        states=states,
        controls=controls,
        feedback_gains=feedback_gains,
        planned_grasp_center_positions=np.asarray(planned_positions),
        planned_arm_obstacle_distances=np.asarray(planned_distances),
        solver_cost_trace=np.asarray(solver_logger.costs, dtype=float),
        solver_stopping_trace=np.asarray(solver_logger.stops, dtype=float),
        initial_position=initial_position,
        initial_rotation=initial_rotation,
        target_position=target_position,
        target_rotation=target_rotation,
        obstacle_center=obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        final_position=final_position,
        final_rotation=final_rotation,
        converged=bool(converged),
        iterations=int(solver.iter),
        cost=float(solver.cost),
        stopping_value=float(solver.stop),
        final_position_error=float(np.linalg.norm(final_position - target_position)),
        final_orientation_error=rotation_distance(final_rotation, target_rotation),
        minimum_arm_obstacle_distance=minimum_distance,
        minimum_arm_obstacle_clearance=(minimum_distance - config.obstacle.safety_margin),
        closest_collision_geometry=closest_geometry,
        final_speed=float(np.linalg.norm(states[-1, 7:])),
        minimum_joint_margin=float(min(np.min(lower_margins), np.min(upper_margins))),
        maximum_torque_ratio=maximum_torque_ratio,
    )
    if validate_solution:
        solution.validate()
    return solution
