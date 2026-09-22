import numpy as np
import pytest

from panda_trajopt.mujoco_sim import (
    ARM_ACTUATOR_NAMES,
    GRASP_CENTER_FROM_HAND,
    PANDA_HOME,
    TABLE_HEIGHT,
    load_panda_simulation,
)


def test_panda_arm_uses_seven_direct_torque_actuators() -> None:
    import mujoco

    simulation = load_panda_simulation()

    assert simulation.model.nq == 9  # seven arm joints and two finger joints
    assert simulation.model.nu == 8  # seven torques and one gripper command
    assert simulation.arm_actuator_ids.shape == (7,)

    for actuator_id, expected_name in zip(
        simulation.arm_actuator_ids, ARM_ACTUATOR_NAMES, strict=True
    ):
        name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_id))
        assert name == expected_name
        assert simulation.model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
        assert simulation.model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_NONE
        assert simulation.model.actuator_gainprm[actuator_id, 0] == 1.0


def test_panda_is_mounted_on_table_without_moving_its_base_frame() -> None:
    import mujoco

    simulation = load_panda_simulation()
    table_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    floor_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    base_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, "link0")

    assert table_id >= 0
    assert simulation.model.geom_pos[table_id, 2] + simulation.model.geom_size[table_id, 2] == 0
    assert simulation.model.geom_pos[floor_id, 2] == -TABLE_HEIGHT
    assert np.allclose(simulation.model.body_pos[base_id], np.zeros(3))


def test_grasp_center_site_is_fixed_between_fingertips() -> None:
    import mujoco

    simulation = load_panda_simulation()
    site_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    hand_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    assert site_id >= 0
    assert simulation.model.site_bodyid[site_id] == hand_id
    assert np.allclose(simulation.model.site_pos[site_id], GRASP_CENTER_FROM_HAND)


def test_obstacle_is_visible_but_does_not_create_unmodelled_contacts() -> None:
    import mujoco

    position = np.array([0.5, 0.04, 0.56])
    simulation = load_panda_simulation(obstacle_position=position, obstacle_radius=0.03)
    obstacle_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "reaching_obstacle")

    assert obstacle_id >= 0
    assert np.allclose(simulation.model.geom_pos[obstacle_id], position)
    assert simulation.model.geom_size[obstacle_id, 0] == pytest.approx(0.03)
    assert simulation.model.geom_contype[obstacle_id] == 0
    assert simulation.model.geom_conaffinity[obstacle_id] == 0
    assert np.isfinite(simulation.minimum_robot_obstacle_distance())


def test_obstacle_requires_both_position_and_radius() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        load_panda_simulation(obstacle_position=[0.5, 0.0, 0.5])


def test_headless_simulation_remains_finite_with_gravity_compensation() -> None:
    simulation = load_panda_simulation()

    for _ in range(10):
        torque = simulation.gravity_compensation_torques()
        applied = simulation.set_arm_torques(torque)
        simulation.step()

    assert np.all(np.isfinite(simulation.data.qpos))
    assert np.all(np.isfinite(simulation.data.qvel))
    assert np.allclose(applied, torque)


def test_arm_state_round_trip_keeps_gripper_separate() -> None:
    simulation = load_panda_simulation()
    gripper_before = simulation.data.qpos[7:].copy()
    velocity = np.linspace(-0.3, 0.3, 7)

    simulation.set_arm_state(PANDA_HOME, velocity)

    assert np.allclose(simulation.arm_configuration, PANDA_HOME)
    assert np.allclose(simulation.arm_velocity, velocity)
    assert np.allclose(simulation.data.qpos[7:], gripper_before)


def test_arm_state_rejects_invalid_shape_and_limits() -> None:
    simulation = load_panda_simulation()

    with pytest.raises(ValueError, match="shapes"):
        simulation.set_arm_state(np.zeros(6), np.zeros(7))
    with pytest.raises(ValueError, match="joint limits"):
        simulation.set_arm_state(np.full(7, 100.0), np.zeros(7))
