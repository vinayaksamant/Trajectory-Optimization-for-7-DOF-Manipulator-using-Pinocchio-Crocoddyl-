#!/usr/bin/env python3
"""Run the first fixed-goal receding-horizon Panda controller headlessly."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.mpc import run_mpc
from panda_trajopt.mujoco_sim import PandaSimulation, load_panda_simulation
from panda_trajopt.reaching import ReachingSolution, reaching_scene_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="Override the number of MPC replans")
    parser.add_argument("--horizon", type=int, help="Override prediction-horizon nodes")
    parser.add_argument("--iterations", type=int, help="Override BoxFDDP iterations per replan")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--start-pause",
        type=float,
        default=1.0,
        help="Seconds to show the initial scene before MPC starts",
    )
    parser.add_argument(
        "--end-pause",
        type=float,
        default=2.0,
        help="Seconds to show the final state after MPC finishes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = (args.steps, args.horizon, args.iterations)
    if any(value is not None and value <= 0 for value in overrides):
        raise SystemExit("MPC overrides must be positive")
    if args.start_pause < 0 or args.end_pause < 0:
        raise SystemExit("Pause durations cannot be negative")

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    config = replace(
        config,
        mpc=replace(
            config.mpc,
            simulation_steps=args.steps or config.mpc.simulation_steps,
            horizon_steps=args.horizon or config.mpc.horizon_steps,
            max_iterations=args.iterations or config.mpc.max_iterations,
        ),
    )
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
        grasp_center_id = simulation.model.site("grasp_center").id
        if viewer is not None:
            from panda_trajopt.visualization import MuJoCoPathOverlay

            viewer.cam.lookat[:] = [0.1, 0.0, 0.35]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            initial_position = simulation.data.site_xpos[grasp_center_id].copy()
            path_overlay = MuJoCoPathOverlay(viewer, initial_position[None, :])
            path_overlay.add_executed_point(initial_position)
            with viewer.lock():
                path_overlay.draw()
            viewer.sync()
            print("MuJoCo viewer ready; MPC starts after the initial pause.")
            time.sleep(args.start_pause)

        def show_replan(step: int, solution: ReachingSolution) -> None:
            if viewer is None:
                return
            if not viewer.is_running():
                raise KeyboardInterrupt
            path_overlay.set_planned_path(solution.planned_grasp_center_positions)
            with viewer.lock():
                path_overlay.draw()
            viewer.sync()
            print(
                f"  replan {step + 1}/{config.mpc.simulation_steps}: "
                f"predicted error={1e3 * solution.final_position_error:.1f} mm"
            )

        def synchronize(current_simulation: PandaSimulation) -> None:
            if viewer is None:
                return
            if not viewer.is_running():
                raise KeyboardInterrupt
            path_overlay.add_executed_point(current_simulation.data.site_xpos[grasp_center_id])
            with viewer.lock():
                path_overlay.draw()
            viewer.sync()
            time.sleep(current_simulation.model.opt.timestep)

        try:
            result = run_mpc(
                panda,
                config,
                simulation,
                on_replan=show_replan,
                after_simulation_step=synchronize,
            )
        except KeyboardInterrupt:
            print("MPC stopped because the viewer was closed.")
            return

        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(args.end_pause)

    print("Fixed-goal BoxFDDP MPC completed")
    print(f"  replans:                    {len(result.replan_times)}")
    print(f"  prediction horizon:         {config.mpc.horizon_steps} nodes")
    print(f"  initial goal error:         {1e3 * result.initial_goal_error:.2f} mm")
    print(f"  final goal error:           {1e3 * result.final_goal_error:.2f} mm")
    print(f"  final orientation error:    {result.final_orientation_error:.3e} rad")
    print(f"  mean replan time:           {1e3 * np.mean(result.replan_times):.1f} ms")
    print(f"  maximum replan time:        {1e3 * np.max(result.replan_times):.1f} ms")
    print(f"  control deadline:           {1e3 * config.trajectory.time_step:.1f} ms")
    print(f"  deadline misses:            {result.deadline_misses}")
    print(f"  converged replans:          {np.count_nonzero(result.solver_converged)}")
    print(f"  minimum safety clearance:  {1e3 * result.minimum_arm_obstacle_clearance:.2f} mm")
    print(f"  maximum torque usage:       {100 * result.maximum_torque_ratio:.1f}%")
    print(f"  torque saturation events:   {result.saturated_control_steps}")
    print("MPC validation: OK")
    if not args.headless:
        print("  viewer paths: current prediction=blue, executed=magenta")


if __name__ == "__main__":
    main()
