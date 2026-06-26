from argparse import ArgumentParser
import time

import numpy as np

from franky import JointImpedanceTracker, Robot


# Franka Panda / FR3 joint limits from the standard libfranka model.
DEFAULT_LOWER_JOINT_LIMITS = [
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

DEFAULT_FRICTION_ENABLED = [1, 1, 1, 1, 1, 1, 1]
DEFAULT_FRICTION_COULOMB = [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2]
DEFAULT_FRICTION_VISCOUS = [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05]
DEFAULT_FRICTION_LIMIT = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def friction_compensation(dq, enabled, coulomb, viscous, torque_limit, epsilon):
    dq = np.asarray(dq, dtype=float)
    enabled = np.asarray(enabled, dtype=bool)
    coulomb = np.asarray(coulomb, dtype=float)
    viscous = np.asarray(viscous, dtype=float)
    torque_limit = np.asarray(torque_limit, dtype=float)

    tau = coulomb * np.tanh(dq / epsilon) + viscous * dq
    tau = np.clip(tau, -torque_limit, torque_limit)
    return np.where(enabled, tau, 0.0)


def format_joint_vector(values):
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument(
        "--stiffness",
        type=float,
        default=2.0,
        help="Joint stiffness",
    )
    parser.add_argument(
        "--lock-joint6",
        action="store_true",
        help="Hold joint 6 at its starting angle",
    )
    parser.add_argument(
        "--lock-joint7",
        action="store_true",
        help="Hold joint 7 at its starting angle",
    )
    parser.add_argument(
        "--lock-stiffness",
        type=float,
        default=40.0,
        help="Stiffness used for locked joints [Nm/rad]",
    )
    parser.add_argument(
        "--friction-enabled",
        type=int,
        nargs=7,
        default=DEFAULT_FRICTION_ENABLED,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Per-joint friction compensation mask, using 0 or 1",
    )
    parser.add_argument(
        "--friction-coulomb",
        type=float,
        nargs=7,
        default=DEFAULT_FRICTION_COULOMB,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Per-joint Coulomb friction compensation [Nm]",
    )
    parser.add_argument(
        "--friction-viscous",
        type=float,
        nargs=7,
        default=DEFAULT_FRICTION_VISCOUS,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Per-joint viscous friction compensation [Nms/rad]",
    )
    parser.add_argument(
        "--friction-limit",
        type=float,
        nargs=7,
        default=DEFAULT_FRICTION_LIMIT,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Per-joint absolute friction compensation torque limit [Nm]",
    )
    parser.add_argument(
        "--friction-mode",
        choices=("cpp", "python", "off"),
        default="cpp",
        help="Where to apply friction compensation",
    )
    parser.add_argument(
        "--friction-epsilon",
        type=float,
        default=0.03,
        help="Velocity scale for tanh smoothing [rad/s]",
    )
    parser.add_argument(
        "--print-period",
        type=float,
        default=0.5,
        help="Seconds between dq/tau_ff diagnostic prints",
    )

    args = parser.parse_args()

    if args.friction_epsilon <= 0.0:
        raise ValueError("--friction-epsilon must be positive")
    if args.print_period < 0.0:
        raise ValueError("--print-period must be non-negative")
    if any(value not in (0, 1) for value in args.friction_enabled):
        raise ValueError("--friction-enabled values must be 0 or 1")
    if any(value < 0.0 for value in args.friction_limit):
        raise ValueError("--friction-limit values must be non-negative")

    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
    )

    q_current = robot.current_joint_positions
    stiffness = [args.stiffness] * 7
    locked_targets = list(q_current)

    if args.lock_joint6:
        stiffness[5] = max(stiffness[5], args.lock_stiffness)
    if args.lock_joint7:
        stiffness[6] = max(stiffness[6], args.lock_stiffness)
    print("Entering compliant joint impedance mode with friction compensation.")
    print(
        "You should be able to guide the robot by hand while feeling pushback near joint limits."
    )
    print(f"Friction mode: {args.friction_mode}")
    print(f"Friction mask: {args.friction_enabled}")
    print(f"Friction Coulomb [Nm]: {format_joint_vector(args.friction_coulomb)}")
    print(f"Friction viscous [Nms/rad]: {format_joint_vector(args.friction_viscous)}")
    print(f"Friction torque limits [Nm]: {format_joint_vector(args.friction_limit)}")
    if args.lock_joint6 or args.lock_joint7:
        locked = []
        if args.lock_joint6:
            locked.append("6")
        if args.lock_joint7:
            locked.append("7")
        print(f"Joint lock active on joint(s): {', '.join(locked)}.")
    print("Press Ctrl-C to stop.")

    friction_enabled = np.asarray(args.friction_enabled, dtype=bool)
    friction_coulomb = np.asarray(args.friction_coulomb, dtype=float)
    friction_viscous = np.asarray(args.friction_viscous, dtype=float)
    friction_limit = np.asarray(args.friction_limit, dtype=float)
    next_print_time = time.perf_counter()

    with JointImpedanceTracker(
        robot,
        stiffness=stiffness,
        compensate_friction=args.friction_mode == "cpp",
        friction_coulomb=friction_enabled * friction_coulomb,
        friction_viscous=friction_enabled * friction_viscous,
        friction_max_torque=friction_enabled * friction_limit,
        friction_velocity_epsilon=args.friction_epsilon,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        period=0.001,
    ) as tracker:
        while tracker.tick():
            state = tracker.state
            q_ref = list(state.q)
            if args.lock_joint6:
                q_ref[5] = locked_targets[5]
            if args.lock_joint7:
                q_ref[6] = locked_targets[6]

            if args.friction_mode == "off":
                friction_preview = np.zeros(7)
            else:
                friction_preview = friction_compensation(
                    state.dq,
                    friction_enabled,
                    friction_coulomb,
                    friction_viscous,
                    friction_limit,
                    args.friction_epsilon,
                )

            if args.friction_mode == "python":
                tau_ff = friction_preview
            else:
                tau_ff = np.zeros(7)
            tracker.set_target(q_ref, dq=[0.0] * 7, tau_ff=tau_ff)

            now = time.perf_counter()
            if args.print_period == 0.0 or now >= next_print_time:
                print(
                    f"dq={format_joint_vector(state.dq)} "
                    f"friction={format_joint_vector(friction_preview)} "
                    f"tau_ff={format_joint_vector(tau_ff)}"
                )
                next_print_time = now + args.print_period
