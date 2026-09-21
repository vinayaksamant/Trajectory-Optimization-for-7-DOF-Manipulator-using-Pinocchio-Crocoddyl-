#!/usr/bin/env python3
"""Check that Pinocchio and MuJoCo describe the same seven-joint Panda."""

from pathlib import Path

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.model_consistency import compare_pinocchio_and_mujoco
from panda_trajopt.mujoco_sim import ARM_JOINT_NAMES, PANDA_HOME, load_panda_simulation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    simulation = load_panda_simulation()

    print("Joint mapping:")
    for index, (pin_name, mujoco_name) in enumerate(
        zip(config.robot.arm_joint_names, ARM_JOINT_NAMES, strict=True), start=1
    ):
        print(f"  {index}: Pinocchio {pin_name:<12} <-> MuJoCo {mujoco_name}")

    report = compare_pinocchio_and_mujoco(panda, simulation, PANDA_HOME)
    print("\nConsistency errors at the shared home pose:")
    print(f"  hand position:    {report.hand_position_error_m:.3e} m")
    print(f"  hand orientation: {report.hand_orientation_error_rad:.3e} rad")
    print(f"  gravity torque:   {report.gravity_max_error_nm:.3e} N m")
    print(f"  joint limits:     {report.joint_limit_max_error_rad:.3e} rad")
    report.validate()
    print("\nmodel consistency: OK")


if __name__ == "__main__":
    main()
