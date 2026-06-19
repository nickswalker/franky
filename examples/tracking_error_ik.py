"""Lissajous tracking error benchmark using IK-backed joint impedance control.

Mirrors tracking_error.py but drives the same trajectory through a damped
least-squares IK solver streaming to a JointImpedanceTracker.  Run both
scripts with identical --speed / --duration flags to compare controllers.
"""

from argparse import ArgumentParser

import numpy as np

from franky import (
    Affine,
    CartesianMotion,
    ControlException,
    Frame,
    FrictionCompensationParams,
    JointImpedanceTracker,
    JointMotion,
    ReferenceType,
    Robot,
)

_DEFAULT_STIFFNESS = np.array([320.0, 320.0, 320.0, 320.0, 120.0, 120.0, 60.0])
_STIFFNESS_RATIOS = _DEFAULT_STIFFNESS / _DEFAULT_STIFFNESS[0]

FRICTION_COULOMB = [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2]
FRICTION_VISCOUS = [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05]

START_JOINTS = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]

CENTER = np.array([0.50, 0.0, 0.40])
AMPLITUDES = np.array([0.08, 0.08, 0.05])
FREQ_RATIOS = np.array([1, 2, 3])
PHASES = np.array([0.0, np.pi / 2, np.pi / 4])


def lissajous(t: float, base_freq: float):
    omega = 2.0 * np.pi * base_freq * FREQ_RATIOS
    pos = CENTER + AMPLITUDES * np.sin(omega * t + PHASES)
    vel = AMPLITUDES * omega * np.cos(omega * t + PHASES)
    return pos, vel


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float
    )
    if theta < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


def rotation_error_rad(R_target: np.ndarray, R_current: np.ndarray) -> float:
    R_err = R_target @ R_current.T
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def get_joint_limits(robot: Robot):
    if "fr3" in robot.model_urdf.lower():
        return (
            np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]),
            np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]),
        )
    return (
        np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    )


class DampedLeastSquaresIK:
    def __init__(
        self,
        robot: Robot,
        lower_joint_limits: np.ndarray,
        upper_joint_limits: np.ndarray,
        *,
        damping: float = 0.05,
        max_iterations: int = 20,
        max_step: float = 0.05,
        position_tolerance: float = 0.002,
        orientation_tolerance: float = 0.03,
    ):
        self.robot = robot
        self.lower_joint_limits = lower_joint_limits
        self.upper_joint_limits = upper_joint_limits
        self.damping = damping
        self.max_iterations = max_iterations
        self.max_step = max_step
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

    def solve(self, target: Affine, seed: np.ndarray):
        q = seed.copy()
        target_matrix = np.asarray(target.matrix, dtype=float)
        target_position = target_matrix[:3, 3]
        target_rotation = target_matrix[:3, :3]
        state = self.robot.state

        for _ in range(self.max_iterations):
            pose = self.robot.model.pose(Frame.EndEffector, q, state.F_T_EE, state.EE_T_K)
            pose_matrix = np.asarray(pose.matrix, dtype=float)
            position_error = target_position - pose_matrix[:3, 3]
            rotation_error = so3_log(target_rotation @ pose_matrix[:3, :3].T)
            error = np.concatenate([position_error, rotation_error])

            if (
                np.linalg.norm(position_error) < self.position_tolerance
                and np.linalg.norm(rotation_error) < self.orientation_tolerance
            ):
                return q, True, error

            jacobian = np.asarray(
                self.robot.model.zero_jacobian(
                    Frame.EndEffector, q, state.F_T_EE, state.EE_T_K
                ),
                dtype=float,
            )
            lhs = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > self.max_step:
                dq *= self.max_step / dq_norm
            q = np.clip(q + dq, self.lower_joint_limits, self.upper_joint_limits)

        return q, False, error


if __name__ == "__main__":
    parser = ArgumentParser(description="Lissajous tracking error benchmark (IK + joint impedance)")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Base frequency of the Lissajous curve in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=_DEFAULT_STIFFNESS[0],
        help="Stiffness of the stiffest joint (joints 1-4) in Nm/rad; "
        "distal joints are scaled proportionally (default: 320)",
    )
    parser.add_argument(
        "--friction",
        action="store_true",
        help="Enable friction compensation",
    )
    parser.add_argument(
        "--inertial-ff",
        action="store_true",
        help="Enable inertial feedforward (M(q) * ddq_d)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Tracking duration in seconds.  Defaults to two full cycles of the "
        "slowest Lissajous frequency.",
    )
    parser.add_argument(
        "--append-tsv",
        metavar="PATH",
        default=None,
        help="Append a result row to this TSV file (creates header if new).",
    )
    args = parser.parse_args()

    base_freq: float = args.speed
    duration: float = args.duration if args.duration is not None else 2.0 / base_freq

    _PERIOD = 0.001

    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
    )
    lower_joint_limits, upper_joint_limits = get_joint_limits(robot)

    # --- move to start configuration -----------------------------------------
    print("Moving to start joint configuration …")
    robot.move(JointMotion(START_JOINTS, reference_type=ReferenceType.Absolute, relative_dynamics_factor=0.1))

    # Start the trajectory at the point of minimum Cartesian velocity so that
    # dq_d[0] ≈ 0
    _t_search = np.arange(0.0, 1.0 / base_freq, 1e-4)
    _vel_norms = np.array([np.linalg.norm(lissajous(t, base_freq)[1]) for t in _t_search])
    t_offset = float(_t_search[np.argmin(_vel_norms)])

    start_pos, _ = lissajous(t_offset, base_freq)
    start_orientation = np.asarray(
        robot.current_pose.end_effector_pose.quaternion, dtype=float
    )
    robot.recover_from_errors()
    try:
        robot.move(
            CartesianMotion(
                Affine(start_pos, start_orientation),
                reference_type=ReferenceType.Absolute,
                relative_dynamics_factor=0.1,
            )
        )
    except ControlException:
        # libfranka sometimes fires a velocity-discontinuity error as the motion
        # generator finishes even though the robot reached the target position.
        robot.recover_from_errors()

    orientation = np.asarray(
        robot.current_pose.end_effector_pose.quaternion, dtype=float
    )
    q_seed = np.asarray(robot.current_joint_positions, dtype=float)

    friction = (
        FrictionCompensationParams(
            coulomb=FRICTION_COULOMB,
            viscous=FRICTION_VISCOUS,
        )
        if args.friction
        else None
    )

    # --- precompute IK trajectory (with disk cache) --------------------------
    import hashlib
    from pathlib import Path

    ik = DampedLeastSquaresIK(robot, lower_joint_limits, upper_joint_limits)

    # Wall-clock sample count; trajectory time is remapped through sinusoidal
    # ease-in/ease-out so ds/dt=0 at both endpoints.
    t_wall = np.arange(0.0, duration + _PERIOD, _PERIOD)
    n = len(t_wall)
    t_norm = np.clip(t_wall / duration, 0.0, 1.0)
    s_norm = t_norm - np.sin(2.0 * np.pi * t_norm) / (2.0 * np.pi)
    t_samples = t_offset + s_norm * duration          # trajectory times fed to lissajous
    # Chain-rule factor mapping f'(t_traj) to the Cartesian velocity x'(t):
    # x'(t) = f'(t_traj) · d(t_traj)/dt, where d(t_traj)/dt = 1 - cos(2π·t_norm).
    dtraj_dt = 1.0 - np.cos(2.0 * np.pi * t_norm)

    state = robot.state
    _jac_damping_sq = 0.05 ** 2

    target_positions = np.array([lissajous(t, base_freq)[0] for t in t_samples])
    _cache_key = hashlib.sha256(target_positions.tobytes()).hexdigest()[:20]
    _cache_path = Path("data/ik_cache") / f"{_cache_key}.npz"

    if _cache_path.exists():
        print(f"Loading cached IK trajectory ({_cache_key}) …")
        _cache = np.load(_cache_path)
        q_traj = _cache["q_traj"]
        dq_traj = _cache["dq_traj"]
        ddq_traj = _cache["ddq_traj"]
        ik_failures = int(_cache["ik_failures"])
    else:
        q_traj = np.zeros((n, 7))
        ik_failures = 0

        print(f"Precomputing IK for {n} samples …")
        q_cur = q_seed.copy()
        ik_trans_errors = []
        ik_rot_errors = []
        for i, t in enumerate(t_samples):
            target_pos = target_positions[i]
            target_pose = Affine(target_pos, orientation)
            q_cur, success, _ = ik.solve(target_pose, q_cur)
            if not success:
                ik_failures += 1
            q_traj[i] = q_cur

            fk_pose = robot.model.pose(Frame.EndEffector, q_cur, state.F_T_EE, state.EE_T_K)
            fk_mat = np.asarray(fk_pose.matrix, dtype=float)
            target_mat = np.asarray(target_pose.matrix, dtype=float)
            ik_trans_errors.append(np.linalg.norm(target_pos - fk_mat[:3, 3]) * 1000.0)
            ik_rot_errors.append(np.degrees(rotation_error_rad(target_mat[:3, :3], fk_mat[:3, :3])))

        if ik_failures:
            print(f"  Warning: IK did not converge for {ik_failures}/{n} samples")
        print(f"  IK theoretical translational error : {float(np.mean(ik_trans_errors)):.2f} mm")
        print(f"  IK theoretical rotational error    : {float(np.mean(ik_rot_errors)):.4f} deg")

        print("Computing analytic joint velocities (J^+ * v_cartesian) …")
        dq_traj = np.zeros((n, 7))
        for i, (t, q) in enumerate(zip(t_samples, q_traj)):
            _, v_linear = lissajous(t, base_freq)
            v_6d = np.concatenate([v_linear * dtraj_dt[i], np.zeros(3)])
            J = np.asarray(
                robot.model.zero_jacobian(Frame.EndEffector, q, state.F_T_EE, state.EE_T_K),
                dtype=float,
            )
            lhs = J @ J.T + _jac_damping_sq * np.eye(6)
            dq_traj[i] = J.T @ np.linalg.solve(lhs, v_6d)

        ddq_traj = np.zeros_like(dq_traj)
        ddq_traj[:-1] = np.diff(dq_traj, axis=0) / _PERIOD

        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_cache_path, q_traj=q_traj, dq_traj=dq_traj, ddq_traj=ddq_traj, ik_failures=np.array(ik_failures))
        print(f"  IK trajectory cached → {_cache_path}")

    tau_ff_traj = np.zeros((n, 7))
    if args.inertial_ff:
        print("Computing inertial feedforward …")
        for i, q in enumerate(q_traj):
            M = np.asarray(
                robot.model.mass(q, state.I_total, state.m_total, state.F_x_Ctotal),
                dtype=float,
            ).reshape(7, 7)
            tau_ff_traj[i] = M @ ddq_traj[i]

    # Precompute target poses for rotational error (orientation is fixed, so
    # any point on the curve has the same target rotation).
    target_mat_fixed = np.asarray(Affine(CENTER, orientation).matrix, dtype=float)

    # Dense curve samples for geometric (nearest-point) translational error.
    _t_curve = np.arange(0.0, 1.0 / base_freq, _PERIOD)
    curve_positions = np.array([lissajous(t, base_freq)[0] for t in _t_curve])

    print(
        f"Tracking Lissajous for {duration:.1f} s  "
        f"(stiffness={args.stiffness:.0f} Nm/rad peak, "
        f"friction={'on' if args.friction else 'off'})"
    )

    trans_errors: list[float] = []
    rot_errors: list[float] = []
    rot_signed_errors: list[np.ndarray] = []
    stiffness = args.stiffness * _STIFFNESS_RATIOS

    with JointImpedanceTracker(
        robot,
        stiffness=stiffness,
        friction=friction,
        period=_PERIOD,
    ) as tracker:
        while tracker.tick():
            t = tracker.elapsed_time
            if t >= duration:
                break

            i = min(int(t / _PERIOD), n - 1)
            tracker.set_target(q_traj[i], dq=dq_traj[i], tau_ff=tau_ff_traj[i])

            current_pose = robot.current_pose.end_effector_pose
            current_pos = np.asarray(current_pose.translation, dtype=float).ravel()
            trans_err = float(np.linalg.norm(curve_positions - current_pos, axis=1).min()) * 1000.0

            current_mat = np.asarray(current_pose.matrix, dtype=float)
            R_err = target_mat_fixed[:3, :3] @ current_mat[:3, :3].T
            rot_err = np.degrees(rotation_error_rad(target_mat_fixed[:3, :3], current_mat[:3, :3]))

            trans_errors.append(trans_err)
            rot_errors.append(rot_err)
            rot_signed_errors.append(np.degrees(so3_log(R_err)))

    te = np.array(trans_errors) if trans_errors else np.zeros(1)
    re = np.array(rot_errors) if rot_errors else np.zeros(1)
    rse = np.array(rot_signed_errors) if rot_signed_errors else np.zeros((1, 3))
    n_samples = len(trans_errors)

    pos_mean = float(np.mean(te))
    pos_rms  = float(np.sqrt(np.mean(te ** 2)))
    pos_max  = float(np.max(te))
    pos_end  = float(te[-1])
    rot_mean = float(np.mean(re))
    rot_rms  = float(np.sqrt(np.mean(re ** 2)))
    rot_max  = float(np.max(re))
    rot_end  = float(re[-1])
    rot_sx, rot_sy, rot_sz = (float(np.mean(rse[:, i])) for i in range(3))

    print(f"\nTracking complete ({n_samples} samples, {ik_failures} IK failures during precompute)")
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
                idx = 1
                while backup.exists():
                    backup = tsv_path.with_suffix(tsv_path.suffix + f".bak{idx}")
                    idx += 1
                tsv_path.rename(backup)
                print(f"  TSV schema changed; archived old file to {backup}")

        write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "episode": "",
            "tracker": "joint_ik",
            "direction": "forward",
            "speed": args.speed,
            "translational_stiffness": args.stiffness,
            "rotational_stiffness": "",
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