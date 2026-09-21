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
    assert config.trajectory.duration == pytest.approx(2.0)


def test_config_rejects_wrong_arm_joint_count(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "panda.yaml").read_text())
    raw["robot"]["arm_joint_names"] = raw["robot"]["arm_joint_names"][:-1]
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="must contain 7 joints"):
        load_config(config_path)

