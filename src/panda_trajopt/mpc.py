"""Receding-horizon BoxFDDP control for the torque-actuated Panda."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from panda_trajopt.config import ProjectConfig
from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PandaSimulation
from panda_trajopt.reaching import ReachingSolution, rotation_distance, solve_reaching_problem


@dataclass(frozen=True)
class MpcResult:
    solver_name: str
    states: np.ndarray
    controls: np.ndarray
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
        if not np.all(np.isfinite(self.states)):
            failures.append("MPC state trace contains non-finite values")
        if not np.all(np.isfinite(self.controls)):
            failures.append("MPC control trace contains non-finite values")
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


def run_mpc(
    panda: PandaModel,
    config: ProjectConfig,
    simulation: PandaSimulation,
    solver_kind: str = "box_fddp",
    on_replan: Callable[[int, ReachingSolution], None] | None = None,
    after_simulation_step: Callable[[PandaSimulation], None] | None = None,
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

    target_position = np.asarray(config.reaching.target_position)
    current_position = simulation.data.site_xpos[grasp_center_id].copy()
    initial_goal_error = float(np.linalg.norm(current_position - target_position))
    torque_limits = np.asarray(config.robot.torque_limits)
    states = [np.concatenate((simulation.arm_configuration, simulation.arm_velocity))]
    controls: list[np.ndarray] = []
    replan_times: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    warm_states: np.ndarray | None = None
    warm_controls: np.ndarray | None = None
    saturated_steps = 0
    maximum_torque_ratio = 0.0
    minimum_obstacle_distance = simulation.minimum_robot_obstacle_distance()
    target_rotation = None

    for _ in range(config.mpc.simulation_steps):
        measured_state = np.concatenate((simulation.arm_configuration, simulation.arm_velocity))
        replan_start = perf_counter()
        solution = solve_reaching_problem(
            panda,
            mpc_config,
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
            on_replan(len(controls), solution)

        command = solution.controls[0]
        applied = simulation.set_arm_torques(command)
        if not np.allclose(applied, command, atol=1e-10, rtol=0.0):
            saturated_steps += 1
        maximum_torque_ratio = max(
            maximum_torque_ratio,
            float(np.max(np.abs(applied) / torque_limits)),
        )
        controls.append(applied.copy())

        for _ in range(substeps):
            simulation.step()
            minimum_obstacle_distance = min(
                minimum_obstacle_distance,
                simulation.minimum_robot_obstacle_distance(),
            )
            if after_simulation_step is not None:
                after_simulation_step(simulation)

        states.append(np.concatenate((simulation.arm_configuration, simulation.arm_velocity)))
        warm_states = np.vstack((solution.states[1:], solution.states[-1]))
        warm_controls = np.vstack((solution.controls[1:], solution.controls[-1]))

    state_array = np.asarray(states)
    lower_margins = state_array[:, :7] - panda.model.lowerPositionLimit
    upper_margins = panda.model.upperPositionLimit - state_array[:, :7]
    final_position = simulation.data.site_xpos[grasp_center_id].copy()
    final_rotation = simulation.data.site_xmat[grasp_center_id].reshape(3, 3).copy()
    if target_rotation is None:
        raise RuntimeError("MPC did not execute any replanning step")
    replan_array = np.asarray(replan_times)
    result = MpcResult(
        solver_name=solver_kind,
        states=state_array,
        controls=np.asarray(controls),
        replan_times=replan_array,
        solver_iterations=np.asarray(iterations),
        solver_converged=np.asarray(converged),
        initial_goal_error=initial_goal_error,
        final_goal_error=float(np.linalg.norm(final_position - target_position)),
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
