"""MuJoCo simulation utilities for the Franka Panda."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"joint{index}_torque" for index in range(1, 8))
PANDA_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TABLE_HEIGHT = 0.75
TABLE_TOP_HALF_SIZE = np.array([0.65, 0.45, 0.03])
TABLE_LEG_HALF_WIDTH = 0.035
GRASP_CENTER_FROM_HAND = np.array([0.0, 0.0, 0.1034])


def default_model_cache() -> Path:
    """Return the project-local cache populated by setup_environment.sh."""
    return Path(__file__).resolve().parents[2] / ".cache" / "mujoco_menagerie"


def _add_table(spec: object) -> None:
    """Place a simple fixed table below the unchanged Panda base frame."""
    import mujoco

    floor = spec.geom("floor")
    if floor is None:
        raise ValueError("The Menagerie scene does not contain the expected floor")
    floor.pos = [0.0, 0.0, -TABLE_HEIGHT]

    top_half_height = float(TABLE_TOP_HALF_SIZE[2])
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, -top_half_height],
        size=TABLE_TOP_HALF_SIZE,
        rgba=[0.45, 0.25, 0.10, 1.0],
        friction=[0.8, 0.02, 0.001],
    )

    leg_half_height = (TABLE_HEIGHT - 2.0 * top_half_height) / 2.0
    leg_center_z = -(2.0 * top_half_height + leg_half_height)
    leg_x = float(TABLE_TOP_HALF_SIZE[0] - 0.10)
    leg_y = float(TABLE_TOP_HALF_SIZE[1] - 0.10)
    for index, (x_position, y_position) in enumerate(
        ((leg_x, leg_y), (leg_x, -leg_y), (-leg_x, leg_y), (-leg_x, -leg_y)),
        start=1,
    ):
        spec.worldbody.add_geom(
            name=f"table_leg_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x_position, y_position, leg_center_z],
            size=[TABLE_LEG_HALF_WIDTH, TABLE_LEG_HALF_WIDTH, leg_half_height],
            rgba=[0.28, 0.14, 0.06, 1.0],
            friction=[0.8, 0.02, 0.001],
        )


def _add_target_marker(
    spec: object,
    target_position: Sequence[float],
    target_rotation: np.ndarray | None = None,
) -> None:
    """Add a non-colliding marker for the Cartesian pose target."""
    import mujoco

    position = np.asarray(target_position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("target_position must contain three finite values")
    rotation = np.eye(3) if target_rotation is None else np.asarray(target_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("target_rotation must be a finite 3-by-3 rotation matrix")
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    spec.worldbody.add_geom(
        name="reaching_target",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=position,
        quat=quaternion,
        size=[0.04, 0.012, 0.006],
        rgba=[0.1, 0.9, 0.2, 0.65],
        contype=0,
        conaffinity=0,
    )


def _add_grasp_center_site(spec: object) -> None:
    """Add the tool-center point midway between the Panda fingertip pads."""
    hand = spec.body("hand")
    if hand is None:
        raise ValueError("The Menagerie Panda model does not contain the 'hand' body")
    hand.add_site(
        name="grasp_center",
        pos=GRASP_CENTER_FROM_HAND,
        size=[0.008],
        rgba=[0.1, 0.4, 1.0, 0.8],
    )


@dataclass
class PandaSimulation:
    """Compiled Panda scene and its mutable MuJoCo state.

    The first seven controls are arm torques in N m. The eighth control belongs
    to the stock position-controlled gripper and is intentionally kept separate.
    """

    model: object
    data: object
    arm_joint_ids: np.ndarray
    arm_qpos_ids: np.ndarray
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

    @property
    def arm_configuration(self) -> np.ndarray:
        """Copy the seven arm joint positions in joint1-to-joint7 order."""
        return self.data.qpos[self.arm_qpos_ids].copy()

    @property
    def arm_velocity(self) -> np.ndarray:
        """Copy the seven arm velocities in joint1-to-joint7 order."""
        return self.data.qvel[self.arm_dof_ids].copy()

    def set_arm_state(self, configuration: Sequence[float], velocity: Sequence[float]) -> None:
        """Set the seven-joint state while leaving the gripper state unchanged."""
        import mujoco

        q = np.asarray(configuration, dtype=float)
        v = np.asarray(velocity, dtype=float)
        if q.shape != (7,) or v.shape != (7,):
            raise ValueError(f"Expected q and v shapes (7,), got q={q.shape}, v={v.shape}")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            raise ValueError("The arm state must contain only finite values")

        limits = self.model.jnt_range[self.arm_joint_ids]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("Arm configuration is outside the Panda joint limits")

        self.data.qpos[self.arm_qpos_ids] = q
        self.data.qvel[self.arm_dof_ids] = v
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


def load_panda_simulation(
    cache_dir: str | Path | None = None,
    target_position: Sequence[float] | None = None,
    target_rotation: np.ndarray | None = None,
) -> PandaSimulation:
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
    _add_table(spec)
    _add_grasp_center_site(spec)
    if target_position is not None:
        _add_target_marker(spec, target_position, target_rotation)
    elif target_rotation is not None:
        raise ValueError("target_rotation requires target_position")

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
    arm_qpos_ids = model.jnt_qposadr[arm_joint_ids].copy()
    simulation = PandaSimulation(
        model, data, arm_joint_ids, arm_qpos_ids, arm_dof_ids, arm_actuator_ids
    )
    simulation.reset_home()
    return simulation
