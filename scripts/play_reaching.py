#!/usr/bin/env python3
"""Optimize the Panda reach and replay it in MuJoCo."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mujoco_sim import load_panda_simulation
from panda_trajopt.playback import replay_reaching_solution
from panda_trajopt.reaching import solve_reaching_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the complete replay without opening a viewer",
    )
    parser.add_argument(
        "--start-pause",
        type=float,
        default=0.75,
        help="Seconds to show the starting pose before motion",
    )
    parser.add_argument(
        "--end-pause",
        type=float,
        default=1.5,
        help="Seconds to show the final pose after motion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_pause < 0 or args.end_pause < 0:
        raise SystemExit("Pause durations cannot be negative")

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    solution = solve_reaching_problem(panda, config)
    simulation = load_panda_simulation(
        target_position=solution.target_position,
        target_rotation=solution.target_rotation,
    )

    if args.headless:
        viewer_manager = nullcontext(None)
    else:
        import mujoco.viewer

        viewer_manager = mujoco.viewer.launch_passive(simulation.model, simulation.data)

    with viewer_manager as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [0.1, 0.0, 0.35]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            viewer.sync()
            time.sleep(args.start_pause)

        def synchronize(_simulation: object) -> None:
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
                time.sleep(simulation.model.opt.timestep)

        try:
            result = replay_reaching_solution(panda, config, solution, simulation, synchronize)
        except KeyboardInterrupt:
            print("Playback stopped because the viewer was closed.")
            return

        if viewer is not None:
            viewer.sync()
            time.sleep(args.end_pause)

    print("Crocoddyl trajectory replayed in MuJoCo")
    print(f"  final grasp-center error: {result.final_grasp_center_error:.3e} m")
    print(f"  final orientation error:  {result.final_grasp_center_orientation_error:.3e} rad")
    print(f"  final joint speed:        {result.final_speed:.3e} rad/s")
    print(f"  RMS joint tracking error: {result.rms_joint_position_error:.3e} rad")
    print(f"  max joint tracking error: {result.maximum_joint_position_error:.3e} rad")
    print(f"  minimum joint margin:     {result.minimum_joint_margin:.3e} rad")
    print(f"  maximum torque usage:     {100 * result.maximum_torque_ratio:.1f}%")
    print(f"  torque saturation events: {result.saturated_control_steps}")
    print("MuJoCo playback: OK")


if __name__ == "__main__":
    main()
