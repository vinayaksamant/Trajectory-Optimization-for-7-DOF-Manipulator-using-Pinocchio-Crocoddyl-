import numpy as np
import pytest

from panda_trajopt.mujoco_sim import ARM_ACTUATOR_NAMES, PANDA_HOME, load_panda_simulation


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
