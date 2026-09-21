"""Numerical checks connecting the Pinocchio and MuJoCo Panda models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_trajopt.model import PandaModel
from panda_trajopt.mujoco_sim import PandaSimulation


@dataclass(frozen=True)
class ModelConsistencyReport:
    hand_position_error_m: float
    hand_orientation_error_rad: float
    gravity_max_error_nm: float
    joint_limit_max_error_rad: float

    def validate(
        self,
        position_tolerance_m: float = 1e-8,
        orientation_tolerance_rad: float = 1e-6,
        gravity_tolerance_nm: float = 1e-5,
        joint_limit_tolerance_rad: float = 1e-8,
    ) -> None:
        checks = {
            "hand position": (self.hand_position_error_m, position_tolerance_m, "m"),
            "hand orientation": (
                self.hand_orientation_error_rad,
                orientation_tolerance_rad,
                "rad",
            ),
            "gravity torque": (self.gravity_max_error_nm, gravity_tolerance_nm, "N m"),
            "joint limits": (
                self.joint_limit_max_error_rad,
                joint_limit_tolerance_rad,
                "rad",
            ),
        }
        failures = [
            f"{name} error {value:.3e} {unit} exceeds {tolerance:.3e} {unit}"
            for name, (value, tolerance, unit) in checks.items()
            if value > tolerance
        ]
        if failures:
            raise ValueError("Pinocchio/MuJoCo model mismatch: " + "; ".join(failures))


def _rotation_error_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def compare_pinocchio_and_mujoco(
    panda: PandaModel,
    simulation: PandaSimulation,
    configuration: np.ndarray,
) -> ModelConsistencyReport:
    """Compare kinematics, gravity, and joint limits at one arm configuration."""
    import mujoco
    import pinocchio as pin

    q = np.asarray(configuration, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"Expected a seven-joint configuration, got shape {q.shape}")

    simulation.set_arm_state(q, np.zeros(7))

    pin_data = panda.model.createData()
    pin.forwardKinematics(panda.model, pin_data, q)
    pin.updateFramePlacements(panda.model, pin_data)
    pin_hand = pin_data.oMf[panda.end_effector_frame_id]

    mujoco_hand_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if mujoco_hand_id < 0:
        raise ValueError("The MuJoCo Panda model does not contain the 'hand' body")
    mujoco_hand_position = simulation.data.xpos[mujoco_hand_id]
    mujoco_hand_rotation = simulation.data.xmat[mujoco_hand_id].reshape(3, 3)

    pin_gravity = pin.computeGeneralizedGravity(panda.model, pin_data, q)
    mujoco_gravity = simulation.gravity_compensation_torques()

    mujoco_joint_limits = simulation.model.jnt_range[simulation.arm_joint_ids]
    pin_joint_limits = np.column_stack(
        (panda.model.lowerPositionLimit, panda.model.upperPositionLimit)
    )

    return ModelConsistencyReport(
        hand_position_error_m=float(np.linalg.norm(pin_hand.translation - mujoco_hand_position)),
        hand_orientation_error_rad=_rotation_error_angle(pin_hand.rotation, mujoco_hand_rotation),
        gravity_max_error_nm=float(np.max(np.abs(pin_gravity - mujoco_gravity))),
        joint_limit_max_error_rad=float(np.max(np.abs(pin_joint_limits - mujoco_joint_limits))),
    )
