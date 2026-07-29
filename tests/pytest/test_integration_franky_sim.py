"""
Integration tests using franky to control the simulated FR3 robot.

Each test spins up a Mujoco-backed SimulationServer, connects with franky's Robot,
executes motions covering every joint / Cartesian direction, and asserts the robot
reached the expected pose.

Torque control is intentionally excluded (out of scope per the task description).
"""

from __future__ import annotations

from contextlib import contextmanager

import franky
import numpy as np
import pytest

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

# Tolerances
JOINT_ATOL = 0.03  # rad – how close the joint must be to the target
CART_ATOL = 0.025  # m   – how close the Cartesian position must be to the target


@contextmanager
def sim_server_context():
    """Start a Genesis simulation + protocol server and tear it down afterwards."""
    with MujocoSimulator() as sim:
        robot = sim.add_robot()
        with SimulationServer(sim) as server:
            server.run_async()
            yield robot


def make_robot(hostname: str) -> franky.Robot:
    """Create a franky Robot connected to the local simulation."""
    return franky.Robot(
        hostname,
        realtime_config=franky.RealtimeConfig.Ignore,
        relative_dynamics_factor=0.2,
    )


# ---------------------------------------------------------------------------
# Test 1 – Joint position control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_position_control():
    """
    Use JointWaypointMotion to drive all 7 joints through four different
    configurations, covering both positive and negative directions for every
    joint.  Verifies that the robot settles at each commanded position.
    """
    # All values are within FR3 joint limits.
    # Initial: [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785]
    waypoints = [
        [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7],
        [0.2, -0.1, 0.1, -1.7, -0.1, 1.6, 0.9],
        [0.0, 0.2, -0.3, -1.6, -0.2, 1.5, 0.3],
        [-0.2, -0.3, 0.2, -1.3, 0.3, 2.0, 1.0],
    ]

    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        for i, waypoint in enumerate(waypoints):
            robot.move(franky.JointWaypointMotion([franky.JointWaypoint(waypoint)]))
            q_actual = list(robot.current_joint_state.position)
            np.testing.assert_allclose(
                q_actual,
                waypoint,
                atol=JOINT_ATOL,
                err_msg=f"Waypoint {i + 1}: joint positions out of tolerance",
            )


# ---------------------------------------------------------------------------
# Test 2 – Joint velocity control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_velocity_control():
    """
    Apply four sequential joint-velocity phases.  Each phase exercises all
    joints; the sign pattern rotates across phases so every joint is driven
    in both directions.  After each phase the sign of the displacement is
    verified.
    """
    hold_ms = 400  # ms per phase

    phases = [
        ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "A – all positive"),
        ([-0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "B – mixed"),
        ([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], "C – alternating"),
        ([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "D – alternating reversed"),
    ]

    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        for velocities, label in phases:
            q_before = np.array(robot.current_joint_state.position)
            robot.move(
                franky.JointVelocityWaypointMotion(
                    [
                        franky.JointVelocityWaypoint(
                            velocities,
                            hold_target_duration=franky.Duration(hold_ms),
                        )
                    ]
                )
            )
            q_after = np.array(robot.current_joint_state.position)
            delta = q_after - q_before

            for i, (v, d) in enumerate(zip(velocities, delta)):
                assert (
                    np.sign(v) == np.sign(d) or abs(d) < 5e-3
                ), f"Phase {label}, joint {i}: commanded v={v:.3f} but Δq={d:.4f}"


# ---------------------------------------------------------------------------
# Test 3 – Cartesian position control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_position_control():
    """
    Move the end-effector to six absolute Cartesian targets that offset the
    initial pose in every world-frame axis direction (±x, ±y, ±z).  Absolute
    targets are used so the commanded translation is expressed in the world
    frame, avoiding the EE-frame ambiguity of relative motions.
    """
    offsets = [
        (np.array([0.05, 0.00, 0.00]), "x+"),
        (np.array([-0.05, 0.00, 0.00]), "x-"),
        (np.array([0.00, 0.04, 0.00]), "y+"),
        (np.array([0.00, -0.04, 0.00]), "y-"),
        (np.array([0.00, 0.00, 0.04]), "z+"),
        (np.array([0.00, 0.00, -0.04]), "z-"),
    ]

    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        for offset, label in offsets:
            target_translation = initial_translation + offset
            target_matrix = initial_pose.matrix.copy()
            target_matrix[:3, 3] = target_translation
            robot.move(
                franky.CartesianMotion(
                    franky.Affine(target_matrix),
                    franky.ReferenceType.Absolute,
                )
            )
            actual_translation = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            ).flatten()
            np.testing.assert_allclose(
                actual_translation,
                target_translation,
                atol=CART_ATOL,
                err_msg=f"Cartesian step {label}: position {actual_translation} != "
                f"expected {target_translation}",
            )


# ---------------------------------------------------------------------------
# Test 4 – Cartesian velocity control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_velocity_control():
    """
    Apply six sequential Cartesian velocity phases covering all three
    translation axes in both directions (+x, −x, +y, −y, +z, −z).
    After each phase the sign of the actual displacement is verified.
    """
    hold_ms = 500

    phases = [
        ([0.02, 0.00, 0.00], "x+"),
        ([-0.02, 0.00, 0.00], "x-"),
        ([0.00, 0.02, 0.00], "y+"),
        ([0.00, -0.02, 0.00], "y-"),
        ([0.00, 0.00, 0.02], "z+"),
        ([0.00, 0.00, -0.02], "z-"),
    ]

    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        for vel_xyz, label in phases:
            pos_before = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            )
            robot.move(
                franky.CartesianVelocityWaypointMotion(
                    [
                        franky.CartesianVelocityWaypoint(
                            franky.Twist(vel_xyz),
                            hold_target_duration=franky.Duration(hold_ms),
                        )
                    ]
                )
            )
            pos_after = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            )
            displacement = pos_after - pos_before

            for axis, (v, d) in enumerate(zip(vel_xyz, displacement)):
                if abs(v) > 1e-6:  # only check commanded axes
                    assert (
                        np.sign(v) == np.sign(d) or abs(d) < 5e-3
                    ), f"Phase {label}, axis {axis}: commanded v={v:.3f} but Δp={d:.4f}"


# ---------------------------------------------------------------------------
# Test 5 - Joint impedance control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_impedance_motion():
    """
    Use JointImpedanceMotion to drive all 7 joints to a target configuration.
    Runs asynchronously and is manually stopped since it does not finish automatically.
    """
    target = [-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8]
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)
        robot.move(franky.JointImpedanceMotion(target), asynchronous=True)

        import time

        time.sleep(1.0)

        robot.stop()
        try:
            robot.join_motion()
        except franky.ControlException as e:
            if "Move command preempted" not in str(e):
                raise

        q_actual = list(robot.current_joint_state.position)
        np.testing.assert_allclose(
            q_actual,
            target,
            atol=JOINT_ATOL,
            err_msg="Joint positions out of tolerance for JointImpedanceMotion",
        )


# ---------------------------------------------------------------------------
# Test 6 - Cartesian impedance control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_impedance_motion():
    """
    Move the end-effector to an absolute Cartesian target using CartesianImpedanceMotion.
    Runs asynchronously and is manually stopped since it does not finish automatically.
    """
    offset = np.array([0.05, 0.05, 0.0])
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)
        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        target_translation = initial_translation + offset
        target_matrix = initial_pose.matrix.copy()
        target_matrix[:3, 3] = target_translation

        robot.move(
            franky.CartesianImpedanceMotion(franky.Affine(target_matrix)),
            asynchronous=True,
        )

        import time

        time.sleep(1.0)

        robot.stop()
        try:
            robot.join_motion()
        except franky.ControlException as e:
            if "Move command preempted" not in str(e):
                raise

        actual_translation = np.array(
            robot.current_cartesian_state.pose.end_effector_pose.translation
        ).flatten()
        np.testing.assert_allclose(
            actual_translation,
            target_translation,
            atol=CART_ATOL,
            err_msg="Position out of tolerance for CartesianImpedanceMotion",
        )


# ---------------------------------------------------------------------------
# Test 6b - Cartesian impedance null-space posture task
# ---------------------------------------------------------------------------

NULLSPACE_ROT_ATOL = 0.1  # rad – how far the held end-effector orientation may drift
NULLSPACE_OFFSET = 0.4 * np.sqrt(2)  # rad – ~0.4 on each of joints 1 and 3


def nullspace_direction(robot: franky.Robot) -> np.ndarray:
    """
    Unit vector spanning the null space of the end-effector Jacobian at the
    robot's current configuration, signed so that joint 1 rotates forward.

    The FR3 has 7 DoF and the Cartesian task constrains 6, so away from
    singularities this is the *only* joint-velocity direction that leaves the
    end-effector pose unchanged.  The controller's projector
    N = I - J^T (J^T)^+ maps every posture torque onto exactly this direction,
    which is what the phases below check from both sides.

    Deriving it from the live Jacobian rather than hard-coding it keeps the
    assertions valid at whatever configuration a phase happens to start from.
    """
    jacobian = np.asarray(
        robot.model.zero_jacobian(franky.Frame.EndEffector, robot.state)
    )
    _, singular_values, vh = np.linalg.svd(jacobian)
    assert singular_values[5] > 1e-2, (
        f"Configuration is near-singular (singular values {singular_values}); the "
        f"null space is not the well-conditioned 1-D space this test assumes"
    )
    direction = vh[6]
    return direction if direction[0] > 0 else -direction


def split_along(vector: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """Split `vector` into its signed component along unit `direction` and the norm of the rest."""
    along = float(vector @ direction)
    return along, float(np.linalg.norm(vector - along * direction))


def max_pose_deviation(states, reference: franky.Affine) -> tuple[float, float]:
    """Largest translational (m) and rotational (rad) deviation from `reference` over `states`."""
    reference_matrix = np.asarray(reference.matrix)
    max_translation = 0.0
    max_rotation = 0.0
    for state in states:
        actual = np.asarray(state.O_T_EE.matrix)
        max_translation = max(
            max_translation,
            float(np.linalg.norm(actual[:3, 3] - reference_matrix[:3, 3])),
        )
        relative = actual[:3, :3] @ reference_matrix[:3, :3].T
        max_rotation = max(
            max_rotation,
            float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))),
        )
    return max_translation, max_rotation


def run_nullspace_posture(
    robot: franky.Robot,
    hold_pose: franky.Affine,
    nullspace_target: np.ndarray,
    nullspace_stiffness,
    duration_s: float = 3.0,
) -> np.ndarray:
    """
    Hold `hold_pose` with a Cartesian impedance motion while a posture task pulls
    the joints toward `nullspace_target`, and assert the end-effector stayed put
    throughout.  Returns the joint displacement over the motion.
    """
    import time

    motion = franky.CartesianImpedanceMotion(
        franky.Affine(hold_pose.matrix),
        nullspace_target=nullspace_target,
        nullspace_stiffness=nullspace_stiffness,
    )
    # Scalars are broadcast to per-joint gains; readback is a 7-vector.
    np.testing.assert_allclose(
        motion.get_nullspace_gains().posture_stiffness,
        np.broadcast_to(nullspace_stiffness, 7),
    )

    states = []
    motion.register_callback(lambda robot_state, *_: states.append(robot_state))

    q_initial = np.array(robot.current_joint_state.position)
    robot.move(motion, asynchronous=True)
    time.sleep(duration_s)
    robot.stop()
    try:
        robot.join_motion()
    except franky.ControlException as e:
        if "Move command preempted" not in str(e):
            raise
    q_final = np.array(robot.current_joint_state.position)

    # Callbacks run on a separate thread and may lag the control loop; let them drain.
    prev_count = -1
    while len(states) != prev_count:
        prev_count = len(states)
        time.sleep(0.2)
    assert len(states) > 100, (
        f"Only {len(states)} control steps recorded for a {duration_s}s motion; "
        f"the end-effector was effectively not monitored"
    )

    # Primary task: the end-effector must hold the commanded pose for the whole
    # motion, not merely be back near it once the arm has settled.  Taking the
    # maximum over every control step is what makes this sensitive to posture
    # torque leaking into the task directions.
    max_translation, max_rotation = max_pose_deviation(states, hold_pose)
    assert max_translation < CART_ATOL, (
        f"Null-space posture task dragged the end-effector: peak translational "
        f"deviation {max_translation:.4f} m over {len(states)} control steps"
    )
    assert max_rotation < NULLSPACE_ROT_ATOL, (
        f"Null-space posture task rotated the end-effector: peak rotational "
        f"deviation {max_rotation:.4f} rad over {len(states)} control steps"
    )

    return q_final - q_initial


@pytest.mark.timeout(60)
def test_cartesian_impedance_nullspace_posture():
    """
    Exercise the null-space posture task of the Cartesian impedance controller.

    The end-effector is commanded to hold its current pose while a secondary
    posture objective (nullspace_target) pulls the joints toward a different
    configuration.  The controller projects the posture torque with
    N = I - J^T (J^T)^+, so it may only move the arm along its one-dimensional
    self-motion manifold.

    The phases constrain that projector from both sides, which is what makes
    them sensitive to it rather than merely to "some posture torque happened":

    * Phase 1 puts stiffness on joint 1 only.  The projected torque vanishes
      exactly when that joint reaches its target, so it must converge there,
      and the displacement must be self-motion (joint 3 counter-rotating), not
      a joint-1 rotation that the Cartesian task then has to fight.
    * Phase 2 offsets the posture target *along* the null-space direction with a
      scalar stiffness: the arm must travel most of that distance, and again
      along the null space.  A projector that zeroed the posture torque would
      leave the arm sitting still here.
    * Phase 3 offsets it by the same amount *orthogonally* to the null space.
      That torque lies in the row space of the Jacobian, so a correct projector
      cancels it exactly and the arm must not move.  An absent or identity
      projector would instead pass it straight through as a pure task-space
      wrench and shove the end-effector.

    Phases 2 and 3 differ only in the direction of an equal-magnitude offset at
    the same stiffness, so the opposite outcomes they require can only come from
    the projection itself.
    """
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        initial_pose = robot.current_cartesian_state.pose.end_effector_pose

        # Phase 1: push just joint 1 toward a target via a per-joint stiffness
        # vector (all other joints have zero stiffness).
        direction = nullspace_direction(robot)
        q_initial = np.array(robot.current_joint_state.position)
        nullspace_target = q_initial.copy()
        nullspace_target[0] += 0.1
        displacement = run_nullspace_posture(
            robot,
            initial_pose,
            nullspace_target,
            np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(
            q_initial[0] + displacement[0],
            nullspace_target[0],
            atol=0.05,
            err_msg="Target joint 1 did not converge to its null-space target",
        )
        # Joint 1 must have got there by self-motion: with the posture torque
        # projected, the counter-rotation of joint 3 comes along with it.  An
        # unprojected joint-1 torque would move joint 1 alone, which has an
        # off-null-space component just as large as its null-space one.
        along, residual = split_along(displacement, direction)
        assert residual < 0.5 * abs(along), (
            f"Posture task did not move the arm along its self-motion manifold: "
            f"{residual:.4f} rad of the displacement is off the null space, "
            f"against {along:.4f} rad along it"
        )

        # Phase 2: offset the posture target along the null-space direction,
        # using a single scalar stiffness.  This is reachable without moving the
        # end-effector, so the arm must travel it.
        direction = nullspace_direction(robot)
        q_initial = np.array(robot.current_joint_state.position)
        displacement = run_nullspace_posture(
            robot, initial_pose, q_initial + NULLSPACE_OFFSET * direction, 20.0
        )
        along, residual = split_along(displacement, direction)
        moved_in_nullspace = along

        # PD control leaves some residual, so require most of the distance.
        assert along > 0.5 * NULLSPACE_OFFSET, (
            f"Joints did not approach the null-space target: covered {along:.4f} rad "
            f"of the commanded {NULLSPACE_OFFSET:.4f} rad along the null space"
        )
        assert residual < 0.3 * along, (
            f"Joints left the self-motion manifold: {residual:.4f} rad of the "
            f"displacement is off the null space, against {along:.4f} rad along it"
        )

        # Phase 3: same stiffness, same offset magnitude, but orthogonal to the
        # null space (at the nominal configuration this is roughly equal, rather
        # than opposite, offsets on joints 1 and 3).  The projector must cancel
        # it, so the arm must stay where it is.
        direction = nullspace_direction(robot)
        q_initial = np.array(robot.current_joint_state.position)
        raw_offset = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        orthogonal_offset = raw_offset - (raw_offset @ direction) * direction
        orthogonal_offset *= NULLSPACE_OFFSET / np.linalg.norm(orthogonal_offset)
        displacement = run_nullspace_posture(
            robot, initial_pose, q_initial + orthogonal_offset, 20.0
        )

        moved = float(np.linalg.norm(displacement))
        assert moved < 0.05, (
            f"An orthogonal posture offset moved the arm by {moved:.4f} rad; its "
            f"torque lies in the row space of the Jacobian and must project to zero"
        )
        assert moved < 0.15 * moved_in_nullspace, (
            f"An orthogonal posture offset moved the arm by {moved:.4f} rad, "
            f"comparable to the {moved_in_nullspace:.4f} rad an equally large "
            f"offset along the null space produced; the posture torque is not "
            f"being projected"
        )


# ---------------------------------------------------------------------------
# Test 7 - Joint impedance tracker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_impedance_tracker():
    """
    Use JointImpedanceTracker to dynamically drive joints to a target configuration.
    """
    target = [-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8]
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)
        with franky.JointImpedanceTracker(robot, period=0.01) as tracker:
            tracker.set_target(target)

            # Tick a few times and wait for it to settle
            for _ in range(100):
                if not tracker.tick():
                    break

        q_actual = list(robot.current_joint_state.position)
        np.testing.assert_allclose(
            q_actual,
            target,
            atol=JOINT_ATOL,
            err_msg="Joint positions out of tolerance for JointImpedanceTracker",
        )


# ---------------------------------------------------------------------------
# Test 8 - Cartesian impedance tracker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_impedance_tracker():
    """
    Move the end-effector to an absolute Cartesian target using CartesianImpedanceTracker.
    """
    offset = np.array([0.05, -0.05, 0.0])
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)
        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        target_translation = initial_translation + offset
        target_matrix = initial_pose.matrix.copy()
        target_matrix[:3, 3] = target_translation

        with franky.CartesianImpedanceTracker(robot, period=0.01) as tracker:
            tracker.set_target(franky.Affine(target_matrix))

            # Tick a few times and wait for it to settle
            for _ in range(100):
                if not tracker.tick():
                    break

        actual_translation = np.array(
            robot.current_cartesian_state.pose.end_effector_pose.translation
        ).flatten()
        np.testing.assert_allclose(
            actual_translation,
            target_translation,
            atol=CART_ATOL,
            err_msg="Position out of tolerance for CartesianImpedanceTracker",
        )


# ---------------------------------------------------------------------------
# Test 9 – Motion callbacks
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_motion_callback_fires_every_time_step():
    """
    Register a motion callback and verify it is invoked once per control step
    of the 1 kHz control loop: a ~5 s motion must produce around 5000
    invocations, in order, with consecutive rel_time values spaced by exactly
    the reported time_step (i.e. no invocation was skipped or dropped).
    """
    hold_ms = 5000

    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        records = []  # (rel_time [ms], time_step [ms]) per invocation

        def cb(robot_state, time_step, rel_time, abs_time, control_signal):
            records.append((rel_time.to_msec(), time_step.to_msec()))

        motion = franky.JointVelocityWaypointMotion(
            [
                franky.JointVelocityWaypoint(
                    [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    hold_target_duration=franky.Duration(hold_ms),
                )
            ]
        )
        motion.register_callback(cb)
        robot.move(motion)

        # Callbacks are queued and run on a separate thread, so they may still
        # be executing after move() returns; wait until the count settles.
        import time

        prev_count = -1
        while len(records) != prev_count:
            prev_count = len(records)
            time.sleep(0.5)

        # One invocation per 1 ms control step: ~5000 for the 5 s hold, plus
        # some slack for the velocity ramp-up/down phases.
        assert hold_ms <= len(records) <= hold_ms + 1000, (
            f"Expected roughly {hold_ms} callback invocations "
            f"(one per 1 kHz control step), got {len(records)}"
        )

        rel_times = np.array([r for r, _ in records])
        time_steps = np.array([s for _, s in records])

        # In order, and no step skipped: each invocation must advance rel_time
        # by exactly the time_step it reported.
        deltas = np.diff(rel_times)
        assert (deltas > 0).all(), "Callback invocations arrived out of order"
        np.testing.assert_array_equal(
            deltas,
            time_steps[1:],
            err_msg="rel_time gaps do not match the reported time steps: "
            "some callback invocations were skipped or dropped",
        )


# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------

GRIPPER_WIDTH_ATOL = 0.005  # m


def make_gripper(hostname: str) -> franky.Gripper:
    """Create a franky Gripper connected to the local simulation gripper server."""
    return franky.Gripper(hostname)


# ---------------------------------------------------------------------------
# Test 5 – Gripper homing
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_homing():
    """
    Home the gripper and verify that it opens to its maximum width.
    """
    with sim_server_context() as robot_server:
        gripper = make_gripper(robot_server.hostname)
        initial_width = gripper.width
        result = gripper.homing()
        assert result, "Gripper homing should return True"
        np.testing.assert_allclose(
            gripper.max_width,
            0.08,
            atol=GRIPPER_WIDTH_ATOL,
            err_msg=(
                f"After homing, max_width {gripper.max_width:.4f} m should equal 0.08m"
            ),
        )
        np.testing.assert_allclose(
            gripper.width,
            initial_width,
            atol=GRIPPER_WIDTH_ATOL,
            err_msg=(
                f"After homing, width {gripper.width:.4f} m should equal "
                f"the prior width {initial_width:.4f} m"
            ),
        )


# ---------------------------------------------------------------------------
# Test 6 – Gripper move
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_move():
    """
    Move the gripper through several target widths and verify the position
    settles at each commanded value.
    """
    target_widths = [0.08, 0.04, 0.01, 0.06, 0.0]

    with sim_server_context() as robot_server:
        gripper = make_gripper(robot_server.hostname)
        for target_width in target_widths:
            result = gripper.move(target_width, 0.05)
            assert result, f"Gripper move to {target_width:.3f} m should return True"
            np.testing.assert_allclose(
                gripper.width,
                target_width,
                atol=GRIPPER_WIDTH_ATOL,
                err_msg=(
                    f"After move({target_width:.3f}), width {gripper.width:.4f} m "
                    f"is outside tolerance"
                ),
            )


# ---------------------------------------------------------------------------
# Test 7 – Gripper grasp success
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_grasp_success():
    """
    Grasp at an achievable width with generous epsilon.  With no physical object
    blocking the gripper, it reaches the commanded width and the epsilon check
    should succeed.
    """
    with sim_server_context() as robot_server:
        gripper = make_gripper(robot_server.hostname)
        gripper.move(0.08, 0.05)  # start fully open

        result = gripper.grasp(0.04, 0.02, 10.0, epsilon_inner=0.02, epsilon_outer=0.02)
        assert result, "Gripper grasp should succeed when width is within epsilon"
        assert gripper.is_grasped, "is_grasped should be True after a successful grasp"


# ---------------------------------------------------------------------------
# Test 8 – Gripper grasp failure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_grasp_failure():
    """
    Grasp at a width slightly above the physical maximum (0.08 m).  The gripper
    can only open to 0.08 m, so it settles there.  The epsilon window around the
    commanded 0.09 m does not cover 0.08 m, so the server returns failure and
    franky raises CommandException.
    """
    with sim_server_context() as robot_server:
        gripper = make_gripper(robot_server.hostname)
        # Commanded 0.09 m, physical limit is 0.08 m:
        # in_range = 0.09 - 0.005 <= 0.08 <= 0.09 + 0.005  →  0.085 <= 0.08  →  False
        with pytest.raises(franky.CommandException):
            gripper.grasp(0.09, 0.02, 10.0, epsilon_inner=0.005, epsilon_outer=0.005)
        assert not gripper.is_grasped, "is_grasped should be False after a failed grasp"


# ---------------------------------------------------------------------------
# Test 9 – Gripper stop
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_stop():
    """
    Issue stop on an idle gripper; the server should reply with kSuccess and
    franky should return True.
    """
    with sim_server_context() as robot_server:
        gripper = make_gripper(robot_server.hostname)
        gripper.homing()
        result = gripper.stop()
        assert result, "Gripper stop should return True"


# ---------------------------------------------------------------------------
# Test 10 – Motion reuse is rejected
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_motion_reuse_raises():
    """
    Motions maintain internal state and must not be executed more than once.
    Reusing a motion object should raise MotionReuseException, both for
    synchronous moves and when preempting an asynchronous motion.
    """
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        motion = franky.JointWaypointMotion(
            [franky.JointWaypoint([-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8])]
        )
        robot.move(motion)
        with pytest.raises(franky.MotionReuseException):
            robot.move(motion)

        # Reuse as a preempting motion of an asynchronous move must also fail.
        async_motion = franky.JointWaypointMotion(
            [franky.JointWaypoint([0.1, -0.1, -0.1, -1.6, -0.1, 1.5, 0.7])]
        )
        robot.move(async_motion, asynchronous=True)
        with pytest.raises(franky.MotionReuseException):
            robot.move(motion, asynchronous=True)
        robot.join_motion()

        # A fresh motion instance still works after the failed attempts.
        robot.move(
            franky.JointWaypointMotion(
                [franky.JointWaypoint([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])]
            )
        )
