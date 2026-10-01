"""Repeatable disturbance and model-mismatch scenarios for Panda MPC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_trajopt.mpc import ExternalTorqueSchedule, MpcResult, TargetSchedule
from panda_trajopt.mujoco_sim import PandaSimulation


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    disturbance_start_step: int | None = None
    disturbance_duration_steps: int = 0
    disturbance_torque: tuple[float, ...] = (0.0,) * 7
    goal_switch_step: int | None = None
    goal_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    payload_mass: float = 0.0
    joint_friction: float = 0.0
    damping_scale: float = 1.0

    def validate(self, simulation_steps: int) -> None:
        if simulation_steps <= 0:
            raise ValueError("simulation_steps must be positive")
        if len(self.disturbance_torque) != 7:
            raise ValueError("disturbance_torque must contain seven values")
        if self.disturbance_start_step is not None:
            if not 0 <= self.disturbance_start_step < simulation_steps:
                raise ValueError("disturbance step must be inside the MPC execution")
            if self.disturbance_duration_steps <= 0:
                raise ValueError("disturbance duration must be positive")
        if self.goal_switch_step is not None and not 0 <= self.goal_switch_step < simulation_steps:
            raise ValueError("goal-switch step must be inside the MPC execution")


SCENARIOS = {
    "disturbance": RobustnessScenario(
        name="disturbance",
        disturbance_start_step=60,
        disturbance_duration_steps=5,
        disturbance_torque=(0.0, 35.0, 0.0, -20.0, 0.0, 0.0, 0.0),
    ),
    "moving_goal": RobustnessScenario(
        name="moving_goal",
        goal_switch_step=40,
        goal_offset=(0.0, -0.08, 0.04),
    ),
    "model_mismatch": RobustnessScenario(
        name="model_mismatch",
        payload_mass=0.75,
        joint_friction=0.15,
        damping_scale=1.3,
    ),
    "all": RobustnessScenario(
        name="all",
        disturbance_start_step=75,
        disturbance_duration_steps=5,
        disturbance_torque=(0.0, 35.0, 0.0, -20.0, 0.0, 0.0, 0.0),
        goal_switch_step=40,
        goal_offset=(0.0, -0.08, 0.04),
        payload_mass=0.75,
        joint_friction=0.15,
        damping_scale=1.3,
    ),
}


@dataclass(frozen=True)
class RobustnessMetrics:
    final_goal_error: float
    peak_goal_error: float
    disturbance_peak_error: float | None
    goal_change_initial_error: float | None
    disturbance_recovery_seconds: float | None
    goal_change_recovery_seconds: float | None

    def validate(self, scenario: RobustnessScenario) -> None:
        failures: list[str] = []
        if self.final_goal_error > 1e-2:
            failures.append(f"final goal error is {self.final_goal_error:.3e} m")
        if (
            scenario.disturbance_start_step is not None
            and self.disturbance_recovery_seconds is None
        ):
            failures.append("controller did not recover after the disturbance")
        if scenario.goal_switch_step is not None and self.goal_change_recovery_seconds is None:
            failures.append("controller did not settle at the changed goal")
        if failures:
            raise ValueError("Robustness checks failed: " + "; ".join(failures))


def configure_simulation_mismatch(
    simulation: PandaSimulation, scenario: RobustnessScenario
) -> None:
    simulation.apply_dynamics_mismatch(
        payload_mass=scenario.payload_mass,
        joint_friction=scenario.joint_friction,
        damping_scale=scenario.damping_scale,
    )


def make_target_schedule(
    initial_target: np.ndarray, scenario: RobustnessScenario
) -> TargetSchedule:
    initial = np.asarray(initial_target, dtype=float).copy()
    changed = initial + np.asarray(scenario.goal_offset)

    def target(step: int) -> np.ndarray:
        if scenario.goal_switch_step is not None and step >= scenario.goal_switch_step:
            return changed
        return initial

    return target


def make_disturbance_schedule(scenario: RobustnessScenario) -> ExternalTorqueSchedule:
    disturbance = np.asarray(scenario.disturbance_torque, dtype=float)

    def external_torque(step: int) -> np.ndarray:
        start = scenario.disturbance_start_step
        if start is not None and start <= step < start + scenario.disturbance_duration_steps:
            return disturbance
        return np.zeros(7)

    return external_torque


def _settling_time(
    errors: np.ndarray,
    start_index: int,
    threshold: float,
    time_step: float,
    hold_nodes: int = 5,
) -> float | None:
    for index in range(start_index, len(errors) - hold_nodes + 1):
        if np.all(errors[index : index + hold_nodes] <= threshold):
            return (index - start_index) * time_step
    return None


def analyze_robustness(
    result: MpcResult,
    scenario: RobustnessScenario,
    time_step: float,
) -> RobustnessMetrics:
    disturbance_recovery = None
    disturbance_peak = None
    if scenario.disturbance_start_step is not None:
        recovery_start = scenario.disturbance_start_step + scenario.disturbance_duration_steps + 1
        pre_error = result.goal_errors[scenario.disturbance_start_step]
        disturbance_recovery = _settling_time(
            result.goal_errors,
            recovery_start,
            max(1e-2, float(pre_error) + 5e-3),
            time_step,
        )
        event_end = min(recovery_start + 20, len(result.goal_errors))
        disturbance_peak = float(
            np.max(result.goal_errors[scenario.disturbance_start_step + 1 : event_end])
        )

    goal_recovery = None
    goal_change_initial_error = None
    if scenario.goal_switch_step is not None:
        goal_change_initial_error = float(result.goal_errors[scenario.goal_switch_step + 1])
        goal_recovery = _settling_time(
            result.goal_errors,
            scenario.goal_switch_step + 1,
            1e-2,
            time_step,
        )

    return RobustnessMetrics(
        final_goal_error=result.final_goal_error,
        peak_goal_error=float(np.max(result.goal_errors)),
        disturbance_peak_error=disturbance_peak,
        goal_change_initial_error=goal_change_initial_error,
        disturbance_recovery_seconds=disturbance_recovery,
        goal_change_recovery_seconds=goal_recovery,
    )
