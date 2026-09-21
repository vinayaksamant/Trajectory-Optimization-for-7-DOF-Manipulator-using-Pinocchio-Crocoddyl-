"""Replay a Crocoddyl Panda trajectory in the independent MuJoCo model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from panda_trajopt.config import ProjectConfig
from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PandaSimulation
from panda_trajopt.reaching import ReachingSolution, rotation_distance


@dataclass(frozen=True)
class PlaybackResult:
    actual_states: np.ndarray
    applied_controls: np.ndarray
    final_grasp_center_position: np.ndarray
    final_grasp_center_rotation: np.ndarray
    final_grasp_center_error: float
    final_grasp_center_orientation_error: float
    final_speed: float
    rms_joint_position_error: float
    maximum_joint_position_error: float
    saturated_control_steps: int
    minimum_joint_margin: float
    maximum_torque_ratio: float

    def validate(self) -> None:
        failures: list[str] = []
        if not np.all(np.isfinite(self.actual_states)):
            failures.append("replayed state contains a non-finite value")
        if not np.all(np.isfinite(self.applied_controls)):
            failures.append("replayed control contains a non-finite value")
        if self.final_grasp_center_error > 1e-2:
            failures.append(
                f"final MuJoCo grasp-center error is {self.final_grasp_center_error:.3e} m"
            )
        if self.final_grasp_center_orientation_error > 1e-2:
            failures.append(
                "final MuJoCo grasp-center orientation error is "
                f"{self.final_grasp_center_orientation_error:.3e} rad"
            )
        if self.final_speed > 5e-2:
            failures.append(f"final MuJoCo joint speed is {self.final_speed:.3e} rad/s")
        if self.minimum_joint_margin < -1e-9:
            failures.append(f"MuJoCo joint limit exceeded by {-self.minimum_joint_margin:.3e} rad")
        if self.maximum_torque_ratio > 1.0 + 1e-9:
            failures.append(f"MuJoCo torque limit ratio is {self.maximum_torque_ratio:.6f}")
        if failures:
            raise ValueError("Invalid MuJoCo playback: " + "; ".join(failures))


def replay_reaching_solution(
    panda: PandaModel,
    config: ProjectConfig,
    solution: ReachingSolution,
    simulation: PandaSimulation,
    after_step: Callable[[PandaSimulation], None] | None = None,
) -> PlaybackResult:
    """Run the discrete Crocoddyl policy at 50 Hz over MuJoCo substeps."""
    import crocoddyl
    import mujoco
    import pinocchio as pin

    optimization_dt = config.trajectory.time_step
    simulation_dt = float(simulation.model.opt.timestep)
    substeps = round(optimization_dt / simulation_dt)
    if substeps <= 0 or not np.isclose(substeps * simulation_dt, optimization_dt):
        raise ValueError(
            f"Crocoddyl dt {optimization_dt} must be an integer multiple of "
            f"MuJoCo dt {simulation_dt}"
        )
    if solution.controls.shape != (config.trajectory.horizon_steps, 7):
        raise ValueError("Reaching controls do not match the configured horizon")
    if solution.feedback_gains.shape != (
        config.trajectory.horizon_steps,
        7,
        14,
    ):
        raise ValueError("Reaching feedback gains have unexpected dimensions")

    state = crocoddyl.StateMultibody(panda.model)
    simulation.reset_home()
    actual_states = [np.concatenate((simulation.arm_configuration, simulation.arm_velocity))]
    applied_controls: list[np.ndarray] = []
    saturated_control_steps = 0
    maximum_torque_ratio = 0.0
    torque_limits = np.asarray(config.robot.torque_limits)

    for node in range(config.trajectory.horizon_steps):
        actual_state = np.concatenate((simulation.arm_configuration, simulation.arm_velocity))
        state_error = state.diff(solution.states[node], actual_state)
        commanded_torque = solution.controls[node] - solution.feedback_gains[node] @ state_error
        applied_torque = simulation.set_arm_torques(commanded_torque)
        if not np.allclose(applied_torque, commanded_torque, atol=1e-10, rtol=0.0):
            saturated_control_steps += 1
        maximum_torque_ratio = max(
            maximum_torque_ratio,
            float(np.max(np.abs(applied_torque) / torque_limits)),
        )
        applied_controls.append(applied_torque.copy())

        for _ in range(substeps):
            simulation.step()
            if after_step is not None:
                after_step(simulation)

        actual_states.append(
            np.concatenate((simulation.arm_configuration, simulation.arm_velocity))
        )

    # The terminal Crocoddyl state has no control associated with it. Use a
    # brief gravity-compensated joint hold to let the independent simulator stop.
    pin_data = panda.model.createData()
    terminal_configuration = solution.states[-1, :7]
    settle_steps = round(config.reaching.playback_settle_time / simulation.model.opt.timestep)
    if not np.isclose(
        settle_steps * simulation.model.opt.timestep,
        config.reaching.playback_settle_time,
    ):
        raise ValueError("playback_settle_time must be a multiple of the MuJoCo timestep")
    for _ in range(settle_steps):
        q = simulation.arm_configuration
        v = simulation.arm_velocity
        gravity = pin.computeGeneralizedGravity(panda.model, pin_data, q)
        commanded_torque = (
            gravity
            - config.reaching.playback_position_gain * (q - terminal_configuration)
            - config.reaching.playback_velocity_gain * v
        )
        applied_torque = simulation.set_arm_torques(commanded_torque)
        if not np.allclose(applied_torque, commanded_torque, atol=1e-10, rtol=0.0):
            saturated_control_steps += 1
        maximum_torque_ratio = max(
            maximum_torque_ratio,
            float(np.max(np.abs(applied_torque) / torque_limits)),
        )
        simulation.step()
        if after_step is not None:
            after_step(simulation)

    actual_states[-1] = np.concatenate((simulation.arm_configuration, simulation.arm_velocity))

    states_array = np.asarray(actual_states)
    controls_array = np.asarray(applied_controls)
    joint_position_errors = states_array[:, :7] - solution.states[:, :7]
    lower_margins = states_array[:, :7] - panda.model.lowerPositionLimit
    upper_margins = panda.model.upperPositionLimit - states_array[:, :7]
    grasp_center_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    if grasp_center_id < 0:
        raise ValueError("The MuJoCo Panda model does not contain the grasp-center site")
    final_grasp_center_position = simulation.data.site_xpos[grasp_center_id].copy()
    final_grasp_center_rotation = simulation.data.site_xmat[grasp_center_id].reshape(3, 3).copy()
    result = PlaybackResult(
        actual_states=states_array,
        applied_controls=controls_array,
        final_grasp_center_position=final_grasp_center_position,
        final_grasp_center_rotation=final_grasp_center_rotation,
        final_grasp_center_error=float(
            np.linalg.norm(final_grasp_center_position - solution.target_position)
        ),
        final_grasp_center_orientation_error=rotation_distance(
            final_grasp_center_rotation, solution.target_rotation
        ),
        final_speed=float(np.linalg.norm(simulation.arm_velocity)),
        rms_joint_position_error=float(np.sqrt(np.mean(joint_position_errors**2))),
        maximum_joint_position_error=float(np.max(np.abs(joint_position_errors))),
        saturated_control_steps=saturated_control_steps,
        minimum_joint_margin=float(min(np.min(lower_margins), np.min(upper_margins))),
        maximum_torque_ratio=maximum_torque_ratio,
    )
    result.validate()
    return result
