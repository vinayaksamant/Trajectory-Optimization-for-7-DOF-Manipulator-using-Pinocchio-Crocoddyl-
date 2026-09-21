from pathlib import Path

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.model_consistency import compare_pinocchio_and_mujoco
from panda_trajopt.mujoco_sim import PANDA_HOME, load_panda_simulation

ROOT = Path(__file__).resolve().parents[1]


def test_pinocchio_and_mujoco_models_match_at_home() -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    simulation = load_panda_simulation()

    report = compare_pinocchio_and_mujoco(panda, simulation, PANDA_HOME)

    report.validate()
