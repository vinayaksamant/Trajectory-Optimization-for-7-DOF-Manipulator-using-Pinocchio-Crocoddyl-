"""Exact whole-arm distance queries used by optimization and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panda_trajopt.model import PandaModel


@dataclass(frozen=True)
class ArmObstacleDistance:
    distance: float
    geometry_name: str


def _distance_with_data(
    panda: PandaModel,
    pin_data: object,
    geometry_data: object,
    configuration: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_geometry: object,
    request: object,
) -> ArmObstacleDistance:
    import coal
    import pinocchio as pin

    pin.updateGeometryPlacements(
        panda.model,
        pin_data,
        panda.collision_model,
        geometry_data,
        configuration,
    )
    obstacle_transform = coal.Transform3s(np.eye(3), obstacle_center)
    minimum_distance = np.inf
    closest_name = ""
    for geometry, placement in zip(
        panda.collision_model.geometryObjects, geometry_data.oMg, strict=True
    ):
        result = coal.DistanceResult()
        robot_transform = coal.Transform3s(placement.rotation, placement.translation)
        distance = coal.distance(
            geometry.geometry,
            robot_transform,
            obstacle_geometry,
            obstacle_transform,
            request,
            result,
        )
        if distance < minimum_distance:
            minimum_distance = float(distance)
            closest_name = geometry.name
    return ArmObstacleDistance(minimum_distance, closest_name)


def minimum_arm_obstacle_distance(
    panda: PandaModel,
    configuration: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> ArmObstacleDistance:
    """Return exact signed surface distance over every Panda collision geometry."""
    import coal

    return _distance_with_data(
        panda,
        panda.model.createData(),
        panda.collision_model.createData(),
        np.asarray(configuration),
        np.asarray(obstacle_center),
        coal.Sphere(obstacle_radius),
        coal.DistanceRequest(),
    )


def make_full_arm_obstacle_residual(
    crocoddyl: object,
    state: object,
    panda: PandaModel,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    activation_distance: float,
    control_dimension: int,
) -> object:
    """Build a Crocoddyl residual for exact minimum whole-arm surface distance."""
    import coal

    center = np.asarray(obstacle_center, dtype=float)

    class FullArmObstacleData(crocoddyl.ResidualDataAbstract):
        def __init__(self, model: object, collector: object) -> None:
            super().__init__(model, collector)
            self.pin_data = panda.model.createData()
            self.geometry_data = panda.collision_model.createData()

    class FullArmObstacleResidual(crocoddyl.ResidualModelAbstract):
        def __init__(self) -> None:
            super().__init__(state, 1, control_dimension, True, False, False)
            self.obstacle_geometry = coal.Sphere(obstacle_radius)
            self.request = coal.DistanceRequest()
            self.finite_difference_step = 1e-6

        def createData(self, collector: object) -> object:
            return FullArmObstacleData(self, collector)

        def distance(self, data: object, configuration: np.ndarray) -> float:
            return _distance_with_data(
                panda,
                data.pin_data,
                data.geometry_data,
                configuration,
                center,
                self.obstacle_geometry,
                self.request,
            ).distance

        def calc(self, data: object, x: np.ndarray, u: np.ndarray | None = None) -> None:
            data.r[0] = self.distance(data, x[: panda.model.nq])

        def calcDiff(self, data: object, x: np.ndarray, u: np.ndarray | None = None) -> None:
            data.Rx[:] = 0.0
            data.Ru[:] = 0.0
            if data.r[0] >= activation_distance:
                return

            step = self.finite_difference_step
            for index in range(panda.model.nv):
                tangent = np.zeros(state.ndx)
                tangent[index] = step
                x_plus = state.integrate(x, tangent)
                x_minus = state.integrate(x, -tangent)
                data.Rx[index] = (
                    self.distance(data, x_plus[: panda.model.nq])
                    - self.distance(data, x_minus[: panda.model.nq])
                ) / (2.0 * step)

    return FullArmObstacleResidual()
