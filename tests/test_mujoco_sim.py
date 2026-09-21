import numpy as np

from panda_trajopt.mujoco_sim import ARM_ACTUATOR_NAMES, load_panda_simulation


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
