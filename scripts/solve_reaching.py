#!/usr/bin/env python3
"""Solve and summarize the first torque-limited Panda reaching problem."""

from pathlib import Path

import numpy as np

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.reaching import solve_reaching_problem


def vector(values: np.ndarray) -> str:
    return np.array2string(values, precision=5, suppress_small=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    solution = solve_reaching_problem(panda, config)

    print(f"Crocoddyl {solution.solver_name} reaching solution")
    print(f"  converged:             {solution.converged}")
    print(f"  iterations:            {solution.iterations}")
    print(f"  duration:              {config.trajectory.duration:.2f} s")
    print(f"  solve time:            {1e3 * solution.solve_time_seconds:.1f} ms")
    print(f"  initial grasp center:  {vector(solution.initial_position)} m")
    print(f"  target grasp center:   {vector(solution.target_position)} m")
    print(f"  final grasp center:    {vector(solution.final_position)} m")
    print(f"  final position error:  {solution.final_position_error:.3e} m")
    print(f"  final rotation error:  {solution.final_orientation_error:.3e} rad")
    print(f"  obstacle center:       {vector(solution.obstacle_center)} m")
    print(f"  obstacle radius:       {solution.obstacle_radius:.3e} m")
    print(f"  safety margin:         {solution.obstacle_safety_margin:.3e} m")
    print(f"  minimum arm gap:       {solution.minimum_arm_obstacle_distance:.3e} m")
    print(f"  minimum clearance:     {solution.minimum_arm_obstacle_clearance:.3e} m")
    print(f"  closest geometry:      {solution.closest_collision_geometry}")
    print(f"  final speed norm:      {solution.final_speed:.3e} rad/s")
    print(f"  minimum joint margin:  {solution.minimum_joint_margin:.3e} rad")
    print(f"  maximum torque usage:  {100 * solution.maximum_torque_ratio:.1f}%")
    print(f"  final cost:            {solution.cost:.6f}")
    print("reaching optimization: OK")


if __name__ == "__main__":
    main()
