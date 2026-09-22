"""Interactive path overlays and post-run trajectory diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from panda_trajopt.config import ProjectConfig
from panda_trajopt.playback import PlaybackResult
from panda_trajopt.reaching import ReachingSolution


class MuJoCoPathOverlay:
    """Draw planned and executed grasp-center paths in a passive viewer."""

    def __init__(
        self,
        viewer: object,
        planned_path: np.ndarray,
        planned_color: np.ndarray | None = None,
    ) -> None:
        self.viewer = viewer
        self.planned_path = np.asarray(planned_path, dtype=float)
        if self.planned_path.ndim != 2 or self.planned_path.shape[1] != 3:
            raise ValueError("planned_path must have shape (N, 3)")
        self.planned_color = (
            np.array([0.1, 0.65, 1.0, 1.0], dtype=np.float32)
            if planned_color is None
            else np.asarray(planned_color, dtype=np.float32)
        )
        if self.planned_color.shape != (4,):
            raise ValueError("planned_color must contain RGBA values")
        self.executed_path: list[np.ndarray] = []

    def add_executed_point(self, position: np.ndarray) -> None:
        point = np.asarray(position, dtype=float).copy()
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("executed path point must contain three finite values")
        if not self.executed_path or np.linalg.norm(point - self.executed_path[-1]) >= 1e-3:
            self.executed_path.append(point)

    def draw(self) -> None:
        """Replace user geoms with blue planned and magenta executed lines."""
        scene = self.viewer.user_scn
        scene.ngeom = 0
        self._draw_path(
            scene,
            self.planned_path,
            self.planned_color,
            3.0,
        )
        self._draw_path(
            scene,
            np.asarray(self.executed_path),
            np.array([1.0, 0.15, 0.65, 1.0], dtype=np.float32),
            4.0,
        )

    @staticmethod
    def _draw_path(
        scene: object,
        points: np.ndarray,
        color: np.ndarray,
        width: float,
    ) -> None:
        import mujoco

        if points.ndim != 2 or len(points) < 2:
            return
        available = scene.maxgeom - scene.ngeom
        if available <= 0:
            return
        stride = max(1, int(np.ceil((len(points) - 1) / available)))
        indices = list(range(0, len(points) - 1, stride))
        if indices[-1] != len(points) - 2:
            indices.append(len(points) - 2)
        for index in indices[:available]:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(-1),
                color,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                points[index],
                points[index + 1],
            )
            scene.ngeom += 1


def _set_equal_3d_axes(axis: object, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_reaching_diagnostics(
    config: ProjectConfig,
    solution: ReachingSolution,
    result: PlaybackResult,
    *,
    show: bool,
    save_path: str | Path | None = None,
) -> object:
    """Plot paths, tracking, torque, clearance, and solver convergence."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solution.validate()
    result.validate()
    planned = solution.planned_grasp_center_positions
    executed = result.actual_grasp_center_positions
    if planned.shape != executed.shape:
        raise ValueError("Planned and executed paths must have matching dimensions")

    state_time = np.arange(len(planned)) * config.trajectory.time_step
    control_time = np.arange(len(solution.controls)) * config.trajectory.time_step
    torque_limits = np.asarray(config.robot.torque_limits)

    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    path_axis = figure.add_subplot(2, 3, 1, projection="3d")
    path_axis.plot(*planned.T, color="tab:blue", label="planned", linewidth=2.0)
    path_axis.plot(*executed.T, color="deeppink", label="executed", linewidth=2.0)
    path_axis.scatter(*solution.target_position, color="limegreen", marker="*", s=100, label="goal")

    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    polar = np.linspace(0.0, np.pi, 16)
    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))
    for radius, color, alpha, label in (
        (config.obstacle.radius, "orangered", 0.35, "obstacle"),
        (
            config.obstacle.radius + config.obstacle.safety_margin,
            "darkorange",
            0.12,
            "safety boundary",
        ),
        (
            config.obstacle.radius + config.obstacle.activation_distance,
            "gold",
            0.06,
            "soft-cost boundary",
        ),
    ):
        path_axis.plot_surface(
            solution.obstacle_center[0] + radius * unit_x,
            solution.obstacle_center[1] + radius * unit_y,
            solution.obstacle_center[2] + radius * unit_z,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label,
        )
    all_path_points = np.vstack((planned, executed, solution.obstacle_center))
    _set_equal_3d_axes(path_axis, all_path_points)
    path_axis.set_title("Grasp-center paths and obstacle regions")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_zlabel("z [m]")
    path_axis.legend(fontsize=8)

    cartesian_axis = figure.add_subplot(2, 3, 2)
    position_error_mm = 1e3 * np.linalg.norm(executed - planned, axis=1)
    cartesian_axis.plot(state_time, position_error_mm, color="tab:red")
    cartesian_axis.set_title("End-effector path tracking error")
    cartesian_axis.set_xlabel("time [s]")
    cartesian_axis.set_ylabel("position error [mm]")
    cartesian_axis.grid(True, alpha=0.3)

    joint_axis = figure.add_subplot(2, 3, 3)
    joint_errors = result.actual_states[:, :7] - solution.states[:, :7]
    for joint in range(7):
        joint_axis.plot(state_time, joint_errors[:, joint], label=f"J{joint + 1}")
    joint_axis.set_title("Joint-position tracking error")
    joint_axis.set_xlabel("time [s]")
    joint_axis.set_ylabel("error [rad]")
    joint_axis.grid(True, alpha=0.3)
    joint_axis.legend(ncol=2, fontsize=8)

    torque_axis = figure.add_subplot(2, 3, 4)
    planned_torque_ratio = 100.0 * solution.controls / torque_limits
    applied_torque_ratio = 100.0 * result.applied_controls / torque_limits
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 7))
    for joint, color in enumerate(colors):
        torque_axis.plot(
            control_time,
            applied_torque_ratio[:, joint],
            color=color,
            label=f"J{joint + 1}",
        )
        torque_axis.plot(
            control_time,
            planned_torque_ratio[:, joint],
            color=color,
            linestyle="--",
            alpha=0.45,
        )
    torque_axis.axhline(100.0, color="black", linestyle=":")
    torque_axis.axhline(-100.0, color="black", linestyle=":")
    torque_axis.set_title("Torque usage: applied solid, planned dashed")
    torque_axis.set_xlabel("time [s]")
    torque_axis.set_ylabel("torque / limit [%]")
    torque_axis.grid(True, alpha=0.3)
    torque_axis.legend(ncol=2, fontsize=8)

    clearance_axis = figure.add_subplot(2, 3, 5)
    clearance_axis.plot(
        state_time,
        1e3 * (solution.planned_arm_obstacle_distances - config.obstacle.safety_margin),
        label="planned (Pinocchio)",
        color="tab:blue",
    )
    clearance_axis.plot(
        state_time,
        1e3 * (result.actual_arm_obstacle_distances - config.obstacle.safety_margin),
        label="executed (MuJoCo)",
        color="deeppink",
    )
    clearance_axis.axhline(0.0, color="red", linestyle="--", label="safety limit")
    clearance_axis.set_title("Full-arm obstacle safety clearance")
    clearance_axis.set_xlabel("time [s]")
    clearance_axis.set_ylabel("clearance above margin [mm]")
    clearance_axis.grid(True, alpha=0.3)
    clearance_axis.legend(fontsize=8)

    convergence_axis = figure.add_subplot(2, 3, 6)
    iterations = np.arange(1, len(solution.solver_cost_trace) + 1)
    convergence_axis.semilogy(
        iterations,
        np.maximum(solution.solver_cost_trace, np.finfo(float).tiny),
        marker="o",
        label="cost",
    )
    convergence_axis.semilogy(
        np.arange(1, len(solution.solver_stopping_trace) + 1),
        np.maximum(solution.solver_stopping_trace, np.finfo(float).tiny),
        marker="s",
        label="stopping value",
    )
    convergence_axis.axhline(
        config.reaching.stopping_threshold,
        color="red",
        linestyle="--",
        label="stopping threshold",
    )
    convergence_axis.set_title("BoxFDDP convergence")
    convergence_axis.set_xlabel("iteration")
    convergence_axis.set_ylabel("log scale")
    convergence_axis.grid(True, which="both", alpha=0.3)
    convergence_axis.legend(fontsize=8)

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
