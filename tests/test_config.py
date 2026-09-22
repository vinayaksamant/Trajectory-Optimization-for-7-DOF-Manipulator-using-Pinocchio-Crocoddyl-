from pathlib import Path

import pytest
import yaml

from panda_trajopt.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_describes_seven_dof_arm() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")

    assert len(config.robot.arm_joint_names) == 7
    assert config.robot.locked_joint_names == (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    assert config.robot.locked_joint_positions == (0.04, 0.04)
    assert config.robot.end_effector_frame == "panda_hand_tcp"
    assert config.robot.torque_limits == (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    assert config.trajectory.duration == pytest.approx(2.0)
    assert len(config.reaching.target_position) == 3
    assert len(config.obstacle.center_position) == 3
    assert config.obstacle.activation_distance == pytest.approx(0.011)


def test_config_rejects_wrong_arm_joint_count(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "panda.yaml").read_text())
    raw["robot"]["arm_joint_names"] = raw["robot"]["arm_joint_names"][:-1]
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="must contain 7 joints"):
        load_config(config_path)


def test_config_rejects_invalid_obstacle_radius(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "panda.yaml").read_text())
    raw["obstacle"]["radius"] = 0.0
    config_path = tmp_path / "bad_obstacle.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="radius must be positive"):
        load_config(config_path)
