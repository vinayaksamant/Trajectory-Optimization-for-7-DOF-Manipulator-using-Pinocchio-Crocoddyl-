#!/usr/bin/env python3
"""Open an interactive MuJoCo viewer for the torque-controlled Panda."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

from panda_trajopt.mujoco_sim import load_panda_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many simulated seconds (default: run until viewer closes)",
    )
    parser.add_argument(
        "--no-gravity-compensation",
        action="store_true",
        help="Apply zero torque so the arm falls instead of holding its home pose",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window; useful for installation checks",
    )
    return parser.parse_args()


def run_simulation(duration: float | None, compensate_gravity: bool, headless: bool) -> None:
    simulation = load_panda_simulation()
    model = simulation.model
    data = simulation.data

    print(f"Loaded Panda: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Arm controls 0..6 are direct joint torques; control 7 operates the gripper.")
    print("Close the viewer window to stop.")

    if headless:
        viewer_manager = nullcontext(None)
    else:
        import mujoco.viewer

        viewer_manager = mujoco.viewer.launch_passive(model, data)

    start_sim_time = data.time
    with viewer_manager as viewer_context:
        while (viewer_context is None or viewer_context.is_running()) and (
            duration is None or data.time - start_sim_time < duration
        ):
            step_started = time.monotonic()
            if compensate_gravity:
                simulation.set_arm_torques(simulation.gravity_compensation_torques())
            else:
                simulation.set_arm_torques([0.0] * 7)

            simulation.step()
            if viewer_context is not None:
                viewer_context.sync()

            remaining = model.opt.timestep - (time.monotonic() - step_started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    run_simulation(args.duration, not args.no_gravity_compensation, args.headless)


if __name__ == "__main__":
    main()
