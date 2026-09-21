"""Configuration types and validation kept independent of robotics libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotConfig:
    name: str
    arm_joint_names: tuple[str, ...]
    locked_joint_names: tuple[str, ...]
    end_effector_frame: str


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
    state_regularization: float
    control_regularization: float


@dataclass(frozen=True)
class ProjectConfig:
    robot: RobotConfig
    trajectory: TrajectoryConfig
    costs: CostConfig


def _positive(data: dict[str, Any], key: str) -> float:
    value = float(data[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Read a YAML file and fail early on the most important mistakes."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    robot = raw["robot"]
    trajectory = raw["trajectory"]
    costs = raw["costs"]

    arm_joint_names = tuple(robot["arm_joint_names"])
    if len(arm_joint_names) != 7:
        raise ValueError(
            f"The Panda arm must contain 7 joints, got {len(arm_joint_names)}"
        )

    horizon_steps = int(trajectory["horizon_steps"])
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    return ProjectConfig(
        robot=RobotConfig(
            name=str(robot["name"]),
            arm_joint_names=arm_joint_names,
            locked_joint_names=tuple(robot["locked_joint_names"]),
            end_effector_frame=str(robot["end_effector_frame"]),
        ),
        trajectory=TrajectoryConfig(
            time_step=_positive(trajectory, "time_step"),
            horizon_steps=horizon_steps,
        ),
        costs=CostConfig(
            end_effector_position=_positive(costs, "end_effector_position"),
            state_regularization=_positive(costs, "state_regularization"),
            control_regularization=_positive(costs, "control_regularization"),
        ),
    )

