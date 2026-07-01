"""Lissajous tracking error benchmark for the Cartesian impedance controller.

Moves to a start joint configuration, then tracks a 3-D Lissajous curve using
the Cartesian impedance tracker.  At the end it reports the average translational
error (mm) and average rotational error (deg).
"""

from argparse import ArgumentParser

import numpy as np

from franky import (
    Affine,
    CartesianImpedanceTracker,
    CartesianMotion,
    FrictionCompensationParams,
    JointMotion,
    ReferenceType,
    Robot,
    Twist,
    TwistAcceleration,
)

# Default friction compensation values (Coulomb, Viscous) tuned for a stock Panda.
_FRICTION_COULOMB = [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2]
_FRICTION_VISCOUS = [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05]

# Joint config that places the EE comfortably in front of the robot at roughly
# (0.50 m, 0.0 m, 0.40 m) with the gripper pointing downward.
_START_JOINTS = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]

# Lissajous geometry (all in metres).
_CENTER = np.array([0.50, 0.0, 0.40])
_AMPLITUDES = np.array([0.10, 0.10, 0.05])
# Integer frequency ratios produce a closed curve.
_FREQ_RATIOS = np.array([1, 2, 3])
_PHASES = np.array([0.0, np.pi / 2, np.pi / 4])


def lissajous(t: float, base_freq: float):
    """Return (position, linear_velocity, linear_acceleration) at time t.

    Velocity and acceleration are derivatives with respect to the function's
    own time argument, i.e. f'(t) and f''(t).
    """
    omega = 2.0 * np.pi * base_freq * _FREQ_RATIOS
    phase = omega * t + _PHASES
    pos = _CENTER + _AMPLITUDES * np.sin(phase)
    vel = _AMPLITUDES * omega * np.cos(phase)
    acc = -_AMPLITUDES * omega ** 2 * np.sin(phase)
    return pos, vel, acc


def total_duration(base_freq: float, n_repeats: float) -> float:
    """Wall-clock duration (s) to traverse n_repeats periods of the base frequency,
    plus a fixed half-cycle ramp up and a fixed half-cycle ramp down."""
    return (n_repeats + 0.5) / base_freq


def trajectory_command(t: float, base_freq: float, n_repeats: float):
    """Position, linear velocity, and linear acceleration commanded at wall-clock time t.

    Ramps dtraj_dt (the rate of trajectory time relative to wall-clock time) from 0 up
    to 1 with a smooth half-cosine profile over a fixed half-cycle of the base
    frequency, holds it at 1 (nominal speed) for a cruise phase, then mirrors the ramp
    back down to 0 -- covering n_repeats periods of the base frequency in total,
    independent of duration. The velocity and acceleration feedforwards follow from
    the chain rule applied to x(t) = lissajous(t_traj(t)):
      x'(t)  = f'(t_traj) * dtraj_dt
      x''(t) = f''(t_traj) * dtraj_dt**2 + f'(t_traj) * d2traj_dt2
    """
    period = 1.0 / base_freq
    ramp_time = 0.5 * period
    cruise_time = n_repeats * period - ramp_time

    if t < ramp_time:
        tau = t
        dtraj_dt = 0.5 * (1.0 - np.cos(np.pi * tau / ramp_time))
        d2traj_dt2 = 0.5 * (np.pi / ramp_time) * np.sin(np.pi * tau / ramp_time)
        t_traj = 0.5 * tau - 0.5 * (ramp_time / np.pi) * np.sin(np.pi * tau / ramp_time)
    elif t < ramp_time + cruise_time:
        tau = t - ramp_time
        dtraj_dt = 1.0
        d2traj_dt2 = 0.0
        t_traj = 0.25 * period + tau
    else:
        tau = t - ramp_time - cruise_time
        dtraj_dt = 0.5 * (1.0 + np.cos(np.pi * tau / ramp_time))
        d2traj_dt2 = -0.5 * (np.pi / ramp_time) * np.sin(np.pi * tau / ramp_time)
        t_traj = 0.25 * period + cruise_time + 0.5 * tau + 0.5 * (ramp_time / np.pi) * np.sin(np.pi * tau / ramp_time)

    pos, vel, acc = lissajous(t_traj, base_freq)
    linear_velocity = vel * dtraj_dt
    linear_acceleration = acc * dtraj_dt ** 2 + vel * d2traj_dt2
    return pos, linear_velocity, linear_acceleration


def max_linear_speed(base_freq: float, n_repeats: float, n_samples: int = 4000) -> float:
    """Peak commanded Cartesian linear speed (m/s) over one run of the trajectory."""
    ts = np.linspace(0.0, total_duration(base_freq, n_repeats), n_samples, endpoint=False)
    speeds = np.array([np.linalg.norm(trajectory_command(t, base_freq, n_repeats)[1]) for t in ts])
    return float(speeds.max())


def velocity_envelope(robot: Robot, n_repeats: float, base_freq_ref: float = 1.0, target_fraction: float = 1.0):
    """Base frequency at which the trajectory's peak Cartesian velocity reaches
    `target_fraction` of the robot's max translational/rotational velocity, whichever
    binds first.
    """
    max_trans_ref = max_linear_speed(base_freq_ref, n_repeats)
    max_rot_ref = 0.0  # constant orientation -> zero angular velocity target

    trans_limit = robot.translation_velocity_limit.max
    rot_limit = robot.rotation_velocity_limit.max

    scale_trans = target_fraction * trans_limit / max_trans_ref if max_trans_ref > 1e-9 else np.inf
    scale_rot = target_fraction * rot_limit / max_rot_ref if max_rot_ref > 1e-9 else np.inf

    if scale_trans <= scale_rot:
        return base_freq_ref * scale_trans, "translational", trans_limit, rot_limit
    return base_freq_ref * scale_rot, "rotational", trans_limit, rot_limit


def rotation_error_rad(R_target: np.ndarray, R_current: np.ndarray) -> float:
    """Geodesic rotation error magnitude in radians."""
    R_err = R_target @ R_current.T
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) from a rotation matrix."""
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    if theta < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


if __name__ == "__main__":
    parser = ArgumentParser(description="Lissajous tracking error benchmark")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Fraction of the Cartesian velocity envelope to run at. 1.0 drives the "
        "trajectory's peak translational/rotational Cartesian velocity (whichever "
        "binds first) to 100%% of the robot's max; 0.5 to 50%%, etc. (default: 0.3)",
    )
    parser.add_argument(
        "--trans-stiff",
        type=float,
        default=1200.0,
        help="Translational stiffness in N/m (default: 1200)",
    )
    parser.add_argument(
        "--rot-stiff",
        type=float,
        default=40.0,
        help="Rotational stiffness in Nm/rad (default: 40)",
    )
    parser.add_argument(
        "--friction",
        action="store_true",
        help="Enable friction compensation",
    )
    parser.add_argument(
        "--inertial-ff",
        action="store_true",
        help="Enable the model-based inertial acceleration feedforward (Lambda(q) * ddx_d)",
    )
    parser.add_argument(
        "--repeats",
        type=float,
        default=2.0,
        help="Number of Lissajous periods (at the base frequency) to traverse, on top "
        "of a fixed half-cycle ramp up and half-cycle ramp down (default: 2.0)",
    )
    parser.add_argument(
        "--append-tsv",
        metavar="PATH",
        default=None,
        help="Append a result row to this TSV file (creates header if new).",
    )
    args = parser.parse_args()
    if args.repeats < 0.5:
        parser.error("--repeats must be at least 0.5 (the ramp up/down alone cover half a cycle each)")

    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
    )

    # Calibrate the envelope on a base_freq_ref=1.0 reference trajectory covering the
    # same --repeats, then let --speed pick a fraction of that envelope. The ramp
    # up/down are fixed at half a cycle of the base frequency, so the trajectory shape
    # in normalized time (t * base_freq) is identical between the reference and the
    # actual run regardless of base_freq -- the peak velocity scales exactly linearly
    # with base_freq and the envelope is hit precisely.
    envelope_base_freq, binding_axis, trans_limit, rot_limit = velocity_envelope(robot, args.repeats)
    base_freq: float = args.speed * envelope_base_freq
    duration: float = total_duration(base_freq, args.repeats)

    print(
        f"Velocity envelope: 100% of {trans_limit:.2f} m/s trans / {rot_limit:.2f} rad/s rot "
        f"reached at base_freq={envelope_base_freq:.4f} Hz (binding: {binding_axis})"
    )
    print(f"speed={args.speed:.2f}x -> base_freq={base_freq:.4f} Hz, duration={duration:.2f} s")

    # --- move to start configuration -----------------------------------------
    print("Moving to start joint configuration …")
    robot.move(JointMotion(_START_JOINTS, reference_type=ReferenceType.Absolute, relative_dynamics_factor=0.1))

    # Move to the first point of the Lissajous at reduced speed so we arrive
    # with a known Cartesian pose (including orientation) before handing off to
    # the impedance tracker.
    start_pos, _, _ = lissajous(0.0, base_freq)
    start_orientation = np.asarray(
        robot.current_pose.end_effector_pose.quaternion, dtype=float
    )
    robot.move(
        CartesianMotion(
            Affine(start_pos, start_orientation),
            reference_type=ReferenceType.Absolute,
            relative_dynamics_factor=0.1,
        )
    )

    friction = (
        FrictionCompensationParams(
            coulomb=_FRICTION_COULOMB,
            viscous=_FRICTION_VISCOUS,
        )
        if args.friction
        else None
    )

    orientation = np.asarray(
        robot.current_pose.end_effector_pose.quaternion, dtype=float
    )

    # Dense curve samples for geometric (nearest-point) translational error.
    _t_curve = np.arange(0.0, 1.0 / base_freq, 0.001)
    curve_positions = np.array([lissajous(t, base_freq)[0] for t in _t_curve])

    print(
        f"Tracking Lissajous for {args.repeats:.2f} cycles + ramps ({duration:.1f} s total)  "
        f"(trans_stiff={args.trans_stiff:.0f} N/m, "
        f"rot_stiff={args.rot_stiff:.0f} Nm/rad, "
        f"friction={'on' if args.friction else 'off'}, "
        f"inertial_ff={'on' if args.inertial_ff else 'off'})"
    )

    trans_errors: list[float] = []
    rot_errors: list[float] = []
    rot_signed_errors: list[np.ndarray] = []

    with CartesianImpedanceTracker(
        robot,
        translational_stiffness=args.trans_stiff,
        rotational_stiffness=args.rot_stiff,
        friction=friction,
        period=0.001,
    ) as tracker:
        while tracker.tick():
            t = tracker.elapsed_time
            if t >= duration:
                break

            target_pos, linear_velocity, linear_acceleration = trajectory_command(t, base_freq, args.repeats)
            target_pose = Affine(target_pos, orientation)
            target_twist = Twist(
                linear_velocity=linear_velocity,
                angular_velocity=np.zeros(3),
            )
            target_acceleration = (
                TwistAcceleration(
                    linear_acceleration=linear_acceleration,
                    angular_acceleration=np.zeros(3),
                )
                if args.inertial_ff
                else None
            )
            tracker.set_target(target_pose, target_twist, target_acceleration)

            current_pose = tracker.current_pose.end_effector_pose
            current_pos = np.asarray(current_pose.translation, dtype=float).ravel()
            trans_err = float(np.linalg.norm(curve_positions - current_pos, axis=1).min()) * 1000.0

            current_mat = np.asarray(current_pose.matrix, dtype=float)
            target_mat = np.asarray(target_pose.matrix, dtype=float)
            R_err = target_mat[:3, :3] @ current_mat[:3, :3].T
            rot_err = np.degrees(rotation_error_rad(target_mat[:3, :3], current_mat[:3, :3]))

            trans_errors.append(trans_err)
            rot_errors.append(rot_err)
            rot_signed_errors.append(np.degrees(so3_log(R_err)))

    te = np.array(trans_errors) if trans_errors else np.zeros(1)
    re = np.array(rot_errors) if rot_errors else np.zeros(1)
    rse = np.array(rot_signed_errors) if rot_signed_errors else np.zeros((1, 3))
    n = len(trans_errors)

    pos_mean  = float(np.mean(te))
    pos_rms   = float(np.sqrt(np.mean(te ** 2)))
    pos_max   = float(np.max(te))
    pos_end   = float(te[-1])
    rot_mean  = float(np.mean(re))
    rot_rms   = float(np.sqrt(np.mean(re ** 2)))
    rot_max   = float(np.max(re))
    rot_end   = float(re[-1])
    rot_sx, rot_sy, rot_sz = (float(np.mean(rse[:, i])) for i in range(3))

    print(f"\nTracking complete ({n} samples)")
    print(f"  pos  mean={pos_mean:.2f}  rms={pos_rms:.2f}  max={pos_max:.2f}  end={pos_end:.2f} mm")
    print(f"  rot  mean={rot_mean:.4f}  rms={rot_rms:.4f}  max={rot_max:.4f}  end={rot_end:.4f} deg")
    print(f"  rot signed  x={rot_sx:.4f}  y={rot_sy:.4f}  z={rot_sz:.4f} deg")

    if args.append_tsv:
        import csv
        import datetime
        from pathlib import Path

        _COLUMNS = [
            "timestamp", "episode", "tracker", "direction",
            "speed", "translational_stiffness", "rotational_stiffness", "nullspace_stiffness",
            "friction_comp", "friction_coulomb", "friction_viscous", "friction_max_torque", "friction_velocity_epsilon",
            "inertial_ff",
            "pos_mean_mm", "pos_rms_mm", "pos_max_mm", "pos_endpoint_mm",
            "rot_mean_deg", "rot_rms_deg", "rot_max_deg", "rot_endpoint_deg",
            "rot_signed_x_deg", "rot_signed_y_deg", "rot_signed_z_deg",
        ]

        def _vec(v):
            return ";".join(f"{x:.6g}" for x in v)

        tsv_path = Path(args.append_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if tsv_path.exists() and tsv_path.stat().st_size > 0:
            with tsv_path.open(newline="") as f:
                existing_header = next(csv.reader(f, delimiter="\t"), [])
            if existing_header != _COLUMNS:
                backup = tsv_path.with_suffix(tsv_path.suffix + ".bak")
                i = 1
                while backup.exists():
                    backup = tsv_path.with_suffix(tsv_path.suffix + f".bak{i}")
                    i += 1
                tsv_path.rename(backup)
                print(f"  TSV schema changed; archived old file to {backup}")

        write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "episode": "",
            "tracker": "cartesian",
            "direction": "forward",
            "speed": args.speed,
            "translational_stiffness": args.trans_stiff,
            "rotational_stiffness": args.rot_stiff,
            "nullspace_stiffness": "",
            "friction_comp": str(args.friction).lower(),
            "friction_coulomb": _vec(friction.coulomb) if friction else "",
            "friction_viscous": _vec(friction.viscous) if friction else "",
            "friction_max_torque": _vec(friction.max_torque) if friction else "",
            "friction_velocity_epsilon": f"{friction.velocity_epsilon:.6g}" if friction else "",
            "inertial_ff": str(args.inertial_ff).lower(),
            "pos_mean_mm": f"{pos_mean:.4f}",
            "pos_rms_mm": f"{pos_rms:.4f}",
            "pos_max_mm": f"{pos_max:.4f}",
            "pos_endpoint_mm": f"{pos_end:.4f}",
            "rot_mean_deg": f"{rot_mean:.6f}",
            "rot_rms_deg": f"{rot_rms:.6f}",
            "rot_max_deg": f"{rot_max:.6f}",
            "rot_endpoint_deg": f"{rot_end:.6f}",
            "rot_signed_x_deg": f"{rot_sx:.6f}",
            "rot_signed_y_deg": f"{rot_sy:.6f}",
            "rot_signed_z_deg": f"{rot_sz:.6f}",
        }
        with tsv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_COLUMNS, delimiter="\t", extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"  Row appended to {tsv_path}")
