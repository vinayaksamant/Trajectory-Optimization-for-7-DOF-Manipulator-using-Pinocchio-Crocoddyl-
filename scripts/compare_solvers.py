#!/usr/bin/env python3
"""Compare Crocoddyl FDDP/DDP-family solvers on one Panda task."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from panda_trajopt.config import load_config
from panda_trajopt.model import load_panda_arm
from panda_trajopt.reaching import ReachingSolution, solve_reaching_problem

SOLVERS = (
    ("box_fddp", "BoxFDDP"),
    ("fddp", "FDDP"),
    ("ilqr", "iLQR-style DDP"),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "panda.yaml",
        help="Task configuration shared by every solver",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override the configured iteration limit for every solver",
    )
    return parser.parse_args()


def validity(solution: ReachingSolution) -> tuple[str, str]:
    try:
        solution.validate()
    except ValueError as error:
        return "NO", str(error).removeprefix("Invalid reaching solution: ")
    return "YES", "all checks passed"


def main() -> None:
    args = parse_args()
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be positive")

    config = load_config(args.config)
    if args.max_iterations is not None:
        config = replace(
            config,
            reaching=replace(config.reaching, max_iterations=args.max_iterations),
        )
    panda = load_panda_arm(config.robot)

    results: list[tuple[str, ReachingSolution, str, str]] = []
    failures: list[tuple[str, str]] = []
    for solver_kind, display_name in SOLVERS:
        try:
            solution = solve_reaching_problem(
                panda,
                config,
                solver_kind=solver_kind,
                validate_solution=False,
            )
        except Exception as error:  # noqa: BLE001 - one solver must not abort the benchmark
            failures.append((display_name, f"{type(error).__name__}: {error}"))
            continue
        valid, reason = validity(solution)
        results.append((display_name, solution, valid, reason))

    print("Identical-task Crocoddyl solver comparison")
    print(f"  config: {args.config}")
    print(
        f"  target: {config.reaching.target_position} m | "
        f"obstacle: {config.obstacle.center_position} m"
    )
    print()
    header = (
        f"{'solver':<17} {'conv':>5} {'iter':>5} {'time ms':>9} "
        f"{'cost':>11} {'pos mm':>9} {'rot mrad':>10} {'torque %':>10} "
        f"{'obs clr mm':>11} {'joint clr':>10} {'valid':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, solution, valid, _ in results:
        print(
            f"{name:<17} {solution.converged!s:>5} {solution.iterations:5d} "
            f"{1e3 * solution.solve_time_seconds:9.1f} {solution.cost:11.4g} "
            f"{1e3 * solution.final_position_error:9.3f} "
            f"{1e3 * solution.final_orientation_error:10.3f} "
            f"{100 * solution.maximum_torque_ratio:10.1f} "
            f"{1e3 * solution.minimum_arm_obstacle_clearance:11.3f} "
            f"{solution.minimum_joint_margin:10.4f} {valid:>6}"
        )

    print("\nConstraint/validation details:")
    for name, _, valid, reason in results:
        print(f"  {name}: {valid} — {reason}")
    for name, reason in failures:
        print(f"  {name}: ERROR — {reason}")

    print("\nInterpretation:")
    print("  BoxFDDP enforces configured torque bounds during optimization.")
    print("  FDDP and iLQR-style SolverDDP are unconstrained baselines;")
    print("  torque ratios above 100% expose that difference.")


if __name__ == "__main__":
    main()
