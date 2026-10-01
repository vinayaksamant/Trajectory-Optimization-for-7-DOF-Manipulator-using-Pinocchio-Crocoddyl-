"""Receding-horizon optimal control for the torque-actuated Panda."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from panda_trajopt.config import ProjectConfig
from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PandaSimulation
from panda_trajopt.reaching import ReachingSolution, rotation_distance, solve_reaching_problem

TargetSchedule = Callable[[int], np.ndarray]
ExternalTorqueSchedule = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class MpcResult:
    solver_name: str
    states: np.ndarray
    controls: np.ndarray
    external_torques: np.ndarray
    grasp_center_positions: np.ndarray
    target_positions: np.ndarray
    goal_errors: np.ndarray
    replan_times: np.ndarray
    solver_iterations: np.ndarray
    solver_converged: np.ndarray
    initial_goal_error: float
    final_goal_error: float
    final_orientation_error: float
    minimum_arm_obstacle_distance: float
    minimum_arm_obstacle_clearance: float
    minimum_joint_margin: float
    maximum_torque_ratio: float
    saturated_control_steps: int
    deadline_misses: int

    def validate(self) -> None:
        failures: list[str] = []
        control_nodes = self.controls.shape[0]
        state_nodes = self.states.shape[0]
        if self.states.ndim != 2 or self.states.shape[1] != 14:
            failures.append("MPC state trace has unexpected dimensions")
        if self.controls.ndim != 2 or self.controls.shape[1] != 7:
            failures.append("MPC control trace has unexpected dimensions")
        if self.external_torques.shape != (control_nodes, 7):
            failures.append("external-torque trace has unexpected dimensions")
        if self.grasp_center_positions.shape != (state_nodes, 3):
            failures.append("grasp-center trace has unexpected dimensions")
        if self.target_positions.shape != (state_nodes, 3):
            failures.append("target-position trace has unexpected dimensions")
        if self.goal_errors.shape != (state_nodes,):
            failures.append("goal-error trace has unexpected dimensions")
        if state_nodes != control_nodes + 1:
            failures.append("MPC state and control traces have inconsistent lengths")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                self.states,
                self.controls,
                self.external_torques,
                self.grasp_center_positions,
                self.target_positions,
                self.goal_errors,
            )
        ):
            failures.append("MPC traces contain non-finite values")
        if self.minimum_arm_obstacle_clearance < -1e-3:
            failures.append(
                "MPC entered the obstacle safety margin by "
                f"{-self.minimum_arm_obstacle_clearance:.3e} m"
            )
        if self.minimum_joint_margin < -1e-9:
            failures.append(f"MPC exceeded a joint limit by {-self.minimum_joint_margin:.3e} rad")
        if self.maximum_torque_ratio > 1.0 + 1e-9:
            failures.append(f"MPC torque limit ratio is {self.maximum_torque_ratio:.6f}")
        if failures:
            raise ValueError("Invalid MPC result: " + "; ".join(failures))


def _scheduled_vector(
    schedule: Callable[[int], np.ndarray] | None,
    step: int,
    default: np.ndarray,
    name: str,
) -> np.ndarray:
    value = default if schedule is None else np.asarray(schedule(step), dtype=float)
    if value.shape != default.shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must return {default.shape[0]} finite values")
    return value.copy()


def run_mpc(
    panda: PandaModel,
    config: ProjectConfig,
    simulation: PandaSimulation,
    solver_kind: str = "box_fddp",
    on_replan: Callable[[int, ReachingSolution], None] | None = None,
    after_simulation_step: Callable[[PandaSimulation], None] | None = None,
    target_position_schedule: TargetSchedule | None = None,
    external_torque_schedule: ExternalTorqueSchedule | None = None,
) -> MpcResult:
    """Replan at every control node and apply only the first optimized torque."""
    import mujoco

    mpc_config = replace(
        config,
        trajectory=replace(config.trajectory, horizon_steps=config.mpc.horizon_steps),
        reaching=replace(config.reaching, max_iterations=config.mpc.max_iterations),
    )
    simulation.reset_home()
    grasp_center_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    if grasp_center_id < 0:
        raise ValueError("The MuJoCo Panda model does not contain the grasp-center site")

    simulation_dt = float(simulation.model.opt.timestep)
    control_dt = config.trajectory.time_step
    substeps = round(control_dt / simulation_dt)
    if substeps <= 0 or not np.isclose(substeps * simulation_dt, control_dt):
        raise ValueError("MPC time step must be an integer multiple of MuJoCo's time step")

    default_target = np.asarray(config.reaching.target_position, dtype=float)
    zero_external_torque = np.zeros(7)
    initial_target = _scheduled_vector(
        target_position_schedule, 0, default_target, "target_position_schedule"
    )
    simulation.set_target_position(initial_target)
    current_position = simulation.data.site_xpos[grasp_center_id].copy()
    initial_goal_error = float(np.linalg.norm(current_position - initial_target))
    torque_limits = np.asarray(config.robot.torque_limits)
    states = [np.concatenate((simulation.arm_configuration, simulation.arm_velocity))]
    grasp_center_positions = [current_position]
    target_positions = [initial_target]
    goal_errors = [initial_goal_error]
    controls: list[np.ndarray] = []
    external_torques: list[np.ndarray] = []
    replan_times: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    warm_states: np.ndarray | None = None
    warm_controls: np.ndarray | None = None
    saturated_steps = 0
    maximum_torque_ratio = 0.0
    minimum_obstacle_distance = simulation.minimum_robot_obstacle_distance()
    target_rotation = None

    for step in range(config.mpc.simulation_steps):
        target_position = _scheduled_vector(
            target_position_schedule, step, default_target, "target_position_schedule"
        )
        step_config = replace(
            mpc_config,
            reaching=replace(
                mpc_config.reaching,
                target_position=tuple(float(value) for value in target_position),
            ),
        )
        simulation.set_target_position(target_position)
        measured_state = np.concatenate((simulation.arm_configuration, simulation.arm_velocity))
        replan_start = perf_counter()
        solution = solve_reaching_problem(
            panda,
            step_config,
            solver_kind=solver_kind,
            validate_solution=False,
            initial_state_override=measured_state,
            warm_start_states=warm_states,
            warm_start_controls=warm_controls,
        )
        replan_times.append(perf_counter() - replan_start)
        iterations.append(solution.iterations)
        converged.append(solution.converged)
        target_rotation = solution.target_rotation
        if on_replan is not None:
            on_replan(step, solution)

        command = solution.controls[0]
        applied = simulation.set_arm_torques(command)
        if not np.allclose(applied, command, atol=1e-10, rtol=0.0):
            saturated_steps += 1
        maximum_torque_ratio = max(
            maximum_torque_ratio,
            float(np.max(np.abs(applied) / torque_limits)),
        )
        controls.append(applied.copy())
        external_torque = _scheduled_vector(
            external_torque_schedule,
            step,
            zero_external_torque,
            "external_torque_schedule",
        )
        external_torques.append(external_torque)
        simulation.set_external_arm_torques(external_torque)

        try:
            for _ in range(substeps):
                simulation.step()
                minimum_obstacle_distance = min(
                    minimum_obstacle_distance,
                    simulation.minimum_robot_obstacle_distance(),
                )
                if after_simulation_step is not None:
                    after_simulation_step(simulation)
        finally:
            simulation.set_external_arm_torques(zero_external_torque)

        states.append(np.concatenate((simulation.arm_configuration, simulation.arm_velocity)))
        current_position = simulation.data.site_xpos[grasp_center_id].copy()
        grasp_center_positions.append(current_position)
        target_positions.append(target_position)
        goal_errors.append(float(np.linalg.norm(current_position - target_position)))
        warm_states = np.vstack((solution.states[1:], solution.states[-1]))
        warm_controls = np.vstack((solution.controls[1:], solution.controls[-1]))

    state_array = np.asarray(states)
    position_array = np.asarray(grasp_center_positions)
    target_array = np.asarray(target_positions)
    error_array = np.asarray(goal_errors)
    lower_margins = state_array[:, :7] - panda.model.lowerPositionLimit
    upper_margins = panda.model.upperPositionLimit - state_array[:, :7]
    final_rotation = simulation.data.site_xmat[grasp_center_id].reshape(3, 3).copy()
    if target_rotation is None:
        raise RuntimeError("MPC did not execute any replanning step")
    replan_array = np.asarray(replan_times)
    result = MpcResult(
        solver_name=solver_kind,
        states=state_array,
        controls=np.asarray(controls),
        external_torques=np.asarray(external_torques),
        grasp_center_positions=position_array,
        target_positions=target_array,
        goal_errors=error_array,
        replan_times=replan_array,
        solver_iterations=np.asarray(iterations),
        solver_converged=np.asarray(converged),
        initial_goal_error=initial_goal_error,
        final_goal_error=float(error_array[-1]),
        final_orientation_error=rotation_distance(final_rotation, target_rotation),
        minimum_arm_obstacle_distance=minimum_obstacle_distance,
        minimum_arm_obstacle_clearance=(minimum_obstacle_distance - config.obstacle.safety_margin),
        minimum_joint_margin=float(min(np.min(lower_margins), np.min(upper_margins))),
        maximum_torque_ratio=maximum_torque_ratio,
        saturated_control_steps=saturated_steps,
        deadline_misses=int(np.count_nonzero(replan_array > control_dt)),
    )
    result.validate()
    return result
