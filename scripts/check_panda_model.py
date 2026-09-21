#!/usr/bin/env python3
"""Load the Panda and print facts we need before implementing optimization."""

from pathlib import Path
import sys

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)

    print(f"robot: {config.robot.name}")
    print(f"arm degrees of freedom: {panda.model.nv}")
    print(f"configuration dimension (nq): {panda.model.nq}")
    print(f"velocity dimension (nv): {panda.model.nv}")
    print(f"end-effector frame: {config.robot.end_effector_frame}")
    print(f"end-effector frame id: {panda.end_effector_frame_id}")
    print(f"planned horizon: {config.trajectory.duration:.2f} s")
    print("model check: OK")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"model check unavailable: {error}", file=sys.stderr)
        raise SystemExit(2) from None
