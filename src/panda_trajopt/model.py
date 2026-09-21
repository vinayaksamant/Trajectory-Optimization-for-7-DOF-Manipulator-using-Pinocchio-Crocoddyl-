"""Load and validate the seven-joint Panda arm model."""

from __future__ import annotations

from dataclasses import dataclass

from panda_trajopt.config import RobotConfig


@dataclass(frozen=True)
class PandaModel:
    """Small bundle used by later kinematics and optimal-control stages."""

    model: object
    collision_model: object
    visual_model: object
    end_effector_frame_id: int


def load_panda_arm(config: RobotConfig) -> PandaModel:
    """Load example-robot-data's Panda and lock the two gripper fingers.

    Imports live inside the function so configuration/tests remain usable before
    the optional robotics environment has been installed.
    """
    try:
        import example_robot_data
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "Pinocchio and example-robot-data are required. Install the "
            "'robotics' optional dependencies described in README.md."
        ) from exc

    robot = example_robot_data.load(config.name)
    model = robot.model
    q_reference = pin.neutral(model)

    missing = [name for name in config.locked_joint_names if not model.existJointName(name)]
    if missing:
        available = ", ".join(model.names)
        raise ValueError(f"Cannot lock missing joints {missing}. Available joints: {available}")

    locked_ids = [model.getJointId(name) for name in config.locked_joint_names]
    model, geometry_models = pin.buildReducedModel(
        model,
        [robot.collision_model, robot.visual_model],
        locked_ids,
        q_reference,
    )
    collision_model, visual_model = geometry_models

    actual_arm_joints = tuple(
        name for name in config.arm_joint_names if model.existJointName(name)
    )
    if actual_arm_joints != config.arm_joint_names or model.nq != 7 or model.nv != 7:
        raise ValueError(
            "The reduced model is not the expected 7-DOF Panda arm: "
            f"nq={model.nq}, nv={model.nv}, joints={actual_arm_joints}"
        )

    if not model.existFrame(config.end_effector_frame):
        frame_names = ", ".join(frame.name for frame in model.frames)
        raise ValueError(
            f"Unknown end-effector frame '{config.end_effector_frame}'. "
            f"Available frames: {frame_names}"
        )

    return PandaModel(
        model=model,
        collision_model=collision_model,
        visual_model=visual_model,
        end_effector_frame_id=model.getFrameId(config.end_effector_frame),
    )

