#!/usr/bin/env python3
"""Run repeatable Panda MPC disturbance and model-mismatch scenarios."""

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
from panda_trajopt.reaching import SOLVER_KINDS, ReachingSolution, reaching_scene_geometry
from panda_trajopt.robustness import (
    SCENARIOS,
    analyze_robustness,
    configure_simulation_mismatch,
    make_disturbance_schedule,
    make_target_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="all")
    parser.add_argument("--solver", choices=SOLVER_KINDS, default="box_fddp")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start-pause", type=float, default=1.0)
    parser.add_argument("--end-pause", type=float, default=2.0)
    return parser.parse_args()


def recovery_text(value: float | None) -> str:
    return "not recovered" if value is None else f"{value:.3f} s"


def main() -> None:
    args = parse_args()
    overrides = (args.steps, args.horizon, args.iterations)
    if any(value is not None and value <= 0 for value in overrides):
        raise SystemExit("MPC overrides must be positive")
    if args.start_pause < 0 or args.end_pause < 0:
        raise SystemExit("Pause durations cannot be negative")

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "panda.yaml")
    config = replace(
        config,
        mpc=replace(
            config.mpc,
            simulation_steps=args.steps or config.mpc.simulation_steps,
            horizon_steps=args.horizon or config.mpc.horizon_steps,
            max_iterations=args.iterations or config.mpc.max_iterations,
        ),
    )
    scenario = SCENARIOS[args.scenario]
    scenario.validate(config.mpc.simulation_steps)
    panda = load_panda_arm(config.robot)
    _, _, initial_target, target_rotation, obstacle_center = reaching_scene_geometry(panda, config)
    simulation = load_panda_simulation(
        target_position=initial_target,
        target_rotation=target_rotation,
        obstacle_position=obstacle_center,
        obstacle_radius=config.obstacle.radius,
        obstacle_safety_margin=config.obstacle.safety_margin,
        obstacle_soft_constraint_buffer=config.obstacle.soft_constraint_buffer,
    )
    configure_simulation_mismatch(simulation, scenario)

    viewer_manager = nullcontext(None)
    if not args.headless:
        import mujoco.viewer

        viewer_manager = mujoco.viewer.launch_passive(simulation.model, simulation.data)

    with viewer_manager as viewer:
        overlay = None
        grasp_center_id = simulation.model.site("grasp_center").id
        if viewer is not None:
            from panda_trajopt.visualization import MuJoCoPathOverlay

            viewer.cam.lookat[:] = [0.1, 0.0, 0.35]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            position = simulation.data.site_xpos[grasp_center_id].copy()
            overlay = MuJoCoPathOverlay(viewer, position[None, :])
            overlay.add_executed_point(position)
            with viewer.lock():
                overlay.draw()
            viewer.sync()
            print(f"Viewer ready for robustness scenario: {scenario.name}")
            time.sleep(args.start_pause)

        def show_replan(step: int, solution: ReachingSolution) -> None:
            if scenario.disturbance_start_step == step:
                print(f"  external torque disturbance starts at step {step}")
            if scenario.goal_switch_step == step:
                print(f"  goal changes at step {step} to {solution.target_position}")
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                overlay.set_planned_path(solution.planned_grasp_center_positions)
                with viewer.lock():
                    overlay.draw()
                viewer.sync()

        def synchronize(current_simulation: PandaSimulation) -> None:
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                overlay.add_executed_point(current_simulation.data.site_xpos[grasp_center_id])
                with viewer.lock():
                    overlay.draw()
                viewer.sync()
                time.sleep(current_simulation.model.opt.timestep)

        try:
            result = run_mpc(
                panda,
                config,
                simulation,
                solver_kind=args.solver,
                on_replan=show_replan,
                after_simulation_step=synchronize,
                target_position_schedule=make_target_schedule(initial_target, scenario),
                external_torque_schedule=make_disturbance_schedule(scenario),
            )
        except KeyboardInterrupt:
            print("Robustness run stopped because the viewer was closed.")
            return
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(args.end_pause)

    metrics = analyze_robustness(result, scenario, config.trajectory.time_step)
    print(f"MPC robustness scenario completed: {scenario.name}")
    print(f"  solver:                    {result.solver_name}")
    print(f"  payload mismatch:          {scenario.payload_mass:.2f} kg")
    print(f"  added joint friction:      {scenario.joint_friction:.2f} Nm")
    print(f"  damping scale:             {scenario.damping_scale:.2f}x")
    print(f"  max external torque:       {np.max(np.abs(result.external_torques)):.2f} Nm")
    print(f"  initial goal error:        {1e3 * result.initial_goal_error:.2f} mm")
    if metrics.disturbance_peak_error is not None:
        print(f"  disturbance peak error:   {1e3 * metrics.disturbance_peak_error:.2f} mm")
    if metrics.goal_change_initial_error is not None:
        print(f"  error after goal change:  {1e3 * metrics.goal_change_initial_error:.2f} mm")
    print(f"  final goal error:          {1e3 * metrics.final_goal_error:.2f} mm")
    print(f"  disturbance recovery:     {recovery_text(metrics.disturbance_recovery_seconds)}")
    print(f"  changed-goal recovery:    {recovery_text(metrics.goal_change_recovery_seconds)}")
    print(f"  minimum safety clearance: {1e3 * result.minimum_arm_obstacle_clearance:.2f} mm")
    print(f"  torque saturation events: {result.saturated_control_steps}")
    print(f"  mean replan time:          {1e3 * np.mean(result.replan_times):.1f} ms")
    print(f"  deadline misses:           {result.deadline_misses}")
    metrics.validate(scenario)
    print("Robustness validation: OK")


if __name__ == "__main__":
    main()
