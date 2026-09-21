"""MuJoCo simulation utilities for the Franka Panda."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"joint{index}_torque" for index in range(1, 8))


def default_model_cache() -> Path:
    """Return the project-local cache populated by setup_environment.sh."""
    return Path(__file__).resolve().parents[2] / ".cache" / "mujoco_menagerie"


@dataclass
class PandaSimulation:
    """Compiled Panda scene and its mutable MuJoCo state.

    The first seven controls are arm torques in N m. The eighth control belongs
    to the stock position-controlled gripper and is intentionally kept separate.
    """

    model: object
    data: object
    arm_dof_ids: np.ndarray
    arm_actuator_ids: np.ndarray

    def reset_home(self) -> None:
        import mujoco

        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise ValueError("The Panda model does not contain the 'home' keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        self.data.ctrl[self.arm_actuator_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_arm_torques(self, torques: Sequence[float]) -> np.ndarray:
        """Apply seven torques after clipping them to the model's safety limits."""
        torque_array = np.asarray(torques, dtype=float)
        if torque_array.shape != (7,):
            raise ValueError(f"Expected 7 arm torques, got shape {torque_array.shape}")

        limits = self.model.actuator_ctrlrange[self.arm_actuator_ids]
        applied = np.clip(torque_array, limits[:, 0], limits[:, 1])
        self.data.ctrl[self.arm_actuator_ids] = applied
        return applied

    def gravity_compensation_torques(self) -> np.ndarray:
        """Return MuJoCo's current bias forces for the seven arm joints."""
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.arm_dof_ids].copy()

    def step(self) -> None:
        import mujoco

        mujoco.mj_step(self.model, self.data)


def load_panda_simulation(cache_dir: str | Path | None = None) -> PandaSimulation:
    """Load the official Menagerie scene and convert its arm to torque control."""
    try:
        import mujoco
        import mujoco_menagerie as menagerie
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo dependencies are missing. Run scripts/setup_environment.sh first."
        ) from exc

    cache = menagerie.Cache(dir=cache_dir or default_model_cache())
    spec = menagerie.get("franka_emika_panda").spec(cache=cache)

    if len(spec.actuators) < 8:
        raise ValueError("Expected seven arm actuators and one gripper actuator")

    for index, (actuator, expected_joint) in enumerate(
        zip(spec.actuators[:7], ARM_JOINT_NAMES, strict=True), start=1
    ):
        if actuator.target != expected_joint:
            raise ValueError(
                f"Actuator {index} targets '{actuator.target}', expected '{expected_joint}'"
            )

        torque_limits = list(actuator.forcerange)
        actuator.set_to_motor()
        actuator.name = ARM_ACTUATOR_NAMES[index - 1]
        actuator.ctrlrange = torque_limits
        actuator.ctrllimited = True
        actuator.forcerange = torque_limits
        actuator.forcelimited = True
        actuator.biasprm = [0.0] * 10

    model = spec.compile()
    data = mujoco.MjData(model)

    arm_actuator_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATOR_NAMES
        ],
        dtype=int,
    )
    arm_joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
        dtype=int,
    )
    if np.any(arm_actuator_ids < 0) or np.any(arm_joint_ids < 0):
        raise ValueError("The compiled Panda model is missing an arm joint or actuator")

    arm_dof_ids = model.jnt_dofadr[arm_joint_ids].copy()
    simulation = PandaSimulation(model, data, arm_dof_ids, arm_actuator_ids)
    simulation.reset_home()
    return simulation
