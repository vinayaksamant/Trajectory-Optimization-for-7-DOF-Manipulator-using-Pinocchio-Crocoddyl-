from pathlib import Path

import numpy as np
import pytest

from panda_trajopt.collision import minimum_arm_obstacle_distance
from panda_trajopt.config import load_config
from panda_trajopt.model import PandaModel, load_panda_arm
from panda_trajopt.mujoco_sim import PANDA_HOME

ROOT = Path(__file__).resolve().parents[1]


def brute_force_distance(
    panda: PandaModel,
    configuration: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> tuple[float, str]:
    import coal
    import pinocchio as pin

    pin_data = panda.model.createData()
    geometry_data = panda.collision_model.createData()
    pin.updateGeometryPlacements(
        panda.model,
        pin_data,
        panda.collision_model,
        geometry_data,
        configuration,
    )
    obstacle = coal.Sphere(obstacle_radius)
    obstacle_transform = coal.Transform3s(np.eye(3), obstacle_center)
    request = coal.DistanceRequest()
    distances = []
    for geometry, placement in zip(
        panda.collision_model.geometryObjects, geometry_data.oMg, strict=True
    ):
        distance = coal.distance(
            geometry.geometry,
            coal.Transform3s(placement.rotation, placement.translation),
            obstacle,
            obstacle_transform,
            request,
            coal.DistanceResult(),
        )
        distances.append((float(distance), geometry.name))
    return min(distances)


@pytest.mark.parametrize(
    "configuration",
    (
        PANDA_HOME,
        PANDA_HOME + np.array([0.1, -0.1, 0.05, 0.1, -0.05, 0.1, 0.05]),
        PANDA_HOME + np.array([-0.2, 0.15, -0.1, -0.15, 0.1, -0.1, -0.1]),
    ),
)
def test_broad_phase_preserves_exact_coal_distance(configuration: np.ndarray) -> None:
    config = load_config(ROOT / "configs" / "panda.yaml")
    panda = load_panda_arm(config.robot)
    obstacle_center = np.asarray(config.obstacle.center_position)

    optimized = minimum_arm_obstacle_distance(
        panda, configuration, obstacle_center, config.obstacle.radius
    )
    brute_distance, brute_name = brute_force_distance(
        panda, configuration, obstacle_center, config.obstacle.radius
    )

    assert optimized.distance == pytest.approx(brute_distance, abs=1e-12)
    assert optimized.geometry_name == brute_name
