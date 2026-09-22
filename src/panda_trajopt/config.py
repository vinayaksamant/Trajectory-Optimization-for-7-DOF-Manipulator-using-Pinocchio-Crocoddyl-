"""Configuration types and validation kept independent of robotics libraries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotConfig:
    name: str
    arm_joint_names: tuple[str, ...]
    locked_joint_names: tuple[str, ...]
    locked_joint_positions: tuple[float, ...]
    end_effector_frame: str
    torque_limits: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryConfig:
    time_step: float
    horizon_steps: int

    @property
    def duration(self) -> float:
        return self.time_step * self.horizon_steps


@dataclass(frozen=True)
class CostConfig:
    end_effector_position: float
    terminal_end_effector_position: float
    end_effector_orientation: float
    terminal_end_effector_orientation: float
    state_regularization: float
    terminal_state_regularization: float
    control_regularization: float
    joint_limits: float
    obstacle_avoidance: float


@dataclass(frozen=True)
class ReachingConfig:
    target_position: tuple[float, float, float]
    target_orientation_rpy: tuple[float, float, float]
    max_iterations: int
    stopping_threshold: float
    playback_settle_time: float
    playback_position_gain: float
    playback_velocity_gain: float


@dataclass(frozen=True)
class ObstacleConfig:
    center_position: tuple[float, float, float]
    radius: float
    safety_margin: float
    soft_constraint_buffer: float

    @property
    def activation_distance(self) -> float:
        return self.safety_margin + self.soft_constraint_buffer


@dataclass(frozen=True)
class ProjectConfig:
    robot: RobotConfig
    trajectory: TrajectoryConfig
    costs: CostConfig
    reaching: ReachingConfig
    obstacle: ObstacleConfig


def _positive(data: dict[str, Any], key: str) -> float:
    value = float(data[key])
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Read a YAML file and fail early on the most important mistakes."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    robot = raw["robot"]
    trajectory = raw["trajectory"]
    costs = raw["costs"]
    reaching = raw["reaching"]
    obstacle = raw["obstacle"]

    arm_joint_names = tuple(robot["arm_joint_names"])
    if len(arm_joint_names) != 7:
        raise ValueError(f"The Panda arm must contain 7 joints, got {len(arm_joint_names)}")

    torque_limits = tuple(float(value) for value in robot["torque_limits"])
    if len(torque_limits) != 7 or any(limit <= 0 for limit in torque_limits):
        raise ValueError("The Panda arm must have 7 positive torque limits")

    locked_joint_positions = tuple(float(value) for value in robot["locked_joint_positions"])
    if len(locked_joint_positions) != len(robot["locked_joint_names"]) or not all(
        isfinite(value) for value in locked_joint_positions
    ):
        raise ValueError("Each locked joint must have one finite locked position")

    horizon_steps = int(trajectory["horizon_steps"])
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    target_position = tuple(float(value) for value in reaching["target_position"])
    if len(target_position) != 3 or not all(isfinite(value) for value in target_position):
        raise ValueError("target_position must contain finite x, y, and z coordinates")
    target_orientation_rpy = tuple(float(value) for value in reaching["target_orientation_rpy"])
    if len(target_orientation_rpy) != 3 or not all(
        isfinite(value) for value in target_orientation_rpy
    ):
        raise ValueError("target_orientation_rpy must contain roll, pitch, and yaw")
    max_iterations = int(reaching["max_iterations"])
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    obstacle_position = tuple(float(value) for value in obstacle["center_position"])
    if len(obstacle_position) != 3 or not all(isfinite(value) for value in obstacle_position):
        raise ValueError("center_position must contain finite x, y, and z coordinates")

    return ProjectConfig(
        robot=RobotConfig(
            name=str(robot["name"]),
            arm_joint_names=arm_joint_names,
            locked_joint_names=tuple(robot["locked_joint_names"]),
            locked_joint_positions=locked_joint_positions,
            end_effector_frame=str(robot["end_effector_frame"]),
            torque_limits=torque_limits,
        ),
        trajectory=TrajectoryConfig(
            time_step=_positive(trajectory, "time_step"),
            horizon_steps=horizon_steps,
        ),
        costs=CostConfig(
            end_effector_position=_positive(costs, "end_effector_position"),
            terminal_end_effector_position=_positive(costs, "terminal_end_effector_position"),
            end_effector_orientation=_positive(costs, "end_effector_orientation"),
            terminal_end_effector_orientation=_positive(costs, "terminal_end_effector_orientation"),
            state_regularization=_positive(costs, "state_regularization"),
            terminal_state_regularization=_positive(costs, "terminal_state_regularization"),
            control_regularization=_positive(costs, "control_regularization"),
            joint_limits=_positive(costs, "joint_limits"),
            obstacle_avoidance=_positive(costs, "obstacle_avoidance"),
        ),
        reaching=ReachingConfig(
            target_position=target_position,
            target_orientation_rpy=target_orientation_rpy,
            max_iterations=max_iterations,
            stopping_threshold=_positive(reaching, "stopping_threshold"),
            playback_settle_time=_positive(reaching, "playback_settle_time"),
            playback_position_gain=_positive(reaching, "playback_position_gain"),
            playback_velocity_gain=_positive(reaching, "playback_velocity_gain"),
        ),
        obstacle=ObstacleConfig(
            center_position=obstacle_position,
            radius=_positive(obstacle, "radius"),
            safety_margin=_positive(obstacle, "safety_margin"),
            soft_constraint_buffer=_positive(obstacle, "soft_constraint_buffer"),
        ),
    )
