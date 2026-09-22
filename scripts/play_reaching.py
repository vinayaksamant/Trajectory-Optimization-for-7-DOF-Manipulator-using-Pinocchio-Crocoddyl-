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
from panda_trajopt.reaching import reaching_scene_geometry, solve_reaching_problem


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open the diagnostics figure after MuJoCo playback",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save the diagnostics figure to this image path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_pause < 0 or args.end_pause < 0:
        raise SystemExit("Pause durations cannot be negative")

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    _, _, target_position, target_rotation, obstacle_center = reaching_scene_geometry(panda, config)
    simulation = load_panda_simulation(
        target_position=target_position,
        target_rotation=target_rotation,
        obstacle_position=obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )

    if args.headless:
        viewer_manager = nullcontext(None)
    else:
        import mujoco.viewer

        viewer_manager = mujoco.viewer.launch_passive(simulation.model, simulation.data)

    with viewer_manager as viewer:
        path_overlay = None
        if viewer is not None:
            grasp_center_id = simulation.model.site("grasp_center").id
            viewer.cam.lookat[:] = [0.1, 0.0, 0.35]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            viewer.sync()
            time.sleep(args.start_pause)

        try:
            solution = solve_reaching_problem(panda, config, validate_solution=False)
        except ValueError as error:
            print(f"Planning configuration rejected: {error}")
            print("No torque commands were executed.")
            if viewer is not None and viewer.is_running():
                time.sleep(args.end_pause)
            return

        validation_error = None
        try:
            solution.validate()
        except ValueError as error:
            validation_error = error

        if viewer is not None:
            if not viewer.is_running():
                print("Playback stopped because the viewer was closed.")
                return
            from panda_trajopt.visualization import MuJoCoPathOverlay

            planned_color = (
                [0.1, 0.65, 1.0, 1.0] if validation_error is None else [1.0, 0.1, 0.1, 1.0]
            )
            path_overlay = MuJoCoPathOverlay(
                viewer,
                solution.planned_grasp_center_positions,
                planned_color=planned_color,
            )
            path_overlay.add_executed_point(simulation.data.site_xpos[grasp_center_id])
            with viewer.lock():
                path_overlay.draw()
            viewer.sync()
            time.sleep(args.start_pause)

        if validation_error is not None:
            print(f"Planned path rejected: {validation_error}")
            print("No torque commands were executed.")
            if viewer is not None:
                print("Rejected planned path is shown in red.")
                time.sleep(args.end_pause)
            return

        print("Planned path accepted; starting torque execution.")

        def synchronize(_simulation: object) -> None:
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                with viewer.lock():
                    path_overlay.add_executed_point(simulation.data.site_xpos[grasp_center_id])
                    path_overlay.draw()
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
    print(f"  minimum arm-obstacle gap:   {result.minimum_arm_obstacle_distance:.3e} m")
    print(f"  minimum safety clearance:  {result.minimum_arm_obstacle_clearance:.3e} m")
    print(f"  final joint speed:        {result.final_speed:.3e} rad/s")
    print(f"  RMS joint tracking error: {result.rms_joint_position_error:.3e} rad")
    print(f"  max joint tracking error: {result.maximum_joint_position_error:.3e} rad")
    print(f"  minimum joint margin:     {result.minimum_joint_margin:.3e} rad")
    print(f"  maximum torque usage:     {100 * result.maximum_torque_ratio:.1f}%")
    print(f"  torque saturation events: {result.saturated_control_steps}")
    print("MuJoCo playback: OK")
    if not args.headless:
        print("  viewer paths: planned=blue, executed=magenta")
        print("  obstacle regions: physical=orange, safety=red, soft-cost=yellow")

    if args.plot or args.save_plot is not None:
        from panda_trajopt.visualization import plot_reaching_diagnostics

        plot_reaching_diagnostics(
            config,
            solution,
            result,
            show=args.plot,
            save_path=args.save_plot,
        )
        if args.save_plot is not None:
            print(f"Diagnostics plot saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
