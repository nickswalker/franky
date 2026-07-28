"""
Unit tests for franky's analytical inverse kinematics (GeoFIK-backed).

These are pure kinematics functions: no robot, sim, or network access is
required. They validate FK/IK round-trips, redundancy handling (swivel and the
joint-angle parameters), the nearest-solution helper, and joint-limit handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from franky import (
    Affine,
    IKOptions,
    RedundancyParameter,
    FR3_JOINT_LIMITS,
    MAX_SWIVEL_SAMPLES,
    MIN_SWIVEL_SAMPLES,
    PANDA_JOINT_LIMITS,
    forward_kinematics,
    inverse_kinematics,
    inverse_kinematics_nearest,
    swivel_angle,
)

# A few configurations comfortably inside the Panda joint limits.
CONFIGS = [
    np.array([0.1, -0.4, 0.2, -1.8, 0.05, 1.5, 0.6]),
    np.array([-0.5, 0.3, 0.7, -1.2, -0.4, 2.0, -0.8]),
    np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.0]),
]

# The analytical redundancy parameters: holding one of these reproduces the source
# configuration exactly. Swivel is resolved by a sampled search and is only
# approximate, so it is tested separately.
ANALYTICAL_PARAMS = [
    RedundancyParameter.Q7,
    RedundancyParameter.Q6,
    RedundancyParameter.Q4,
]

ALL_PARAMS = ANALYTICAL_PARAMS + [RedundancyParameter.Swivel]

# Flange poses for CONFIGS[0..2] as computed by libfranka's own model
# (franka::Model::pose(Frame::kFlange), pinocchio-backed, from libfranka's FR3 URDF),
# column-major 4x4. franky's analytical FK uses the nominal DH parameters instead, so
# this pins the two implementations together.
LIBFRANKA_FLANGE_POSES = [
    np.array(
        [9.491092787709491e-01, -3.012129688146091e-01, 9.199089285781840e-02, 0.0,
         -2.998251765711921e-01, -9.535568775449027e-01, -2.888156472335445e-02, 0.0,
         9.641805041040193e-02, -1.694246296775770e-04, -9.953409118740935e-01, 0.0,
         3.917782867941731e-01, 1.502304121987587e-01, 7.006754092808227e-01, 1.0]
    ).reshape((4, 4)).T,
    np.array(
        [4.876224356831947e-01, 8.445771607595851e-01, 2.211645987532062e-01, 0.0,
         6.740165108738873e-01, -5.251866811890351e-01, 5.195004263434612e-01, 0.0,
         5.549108967102554e-01, -1.042514720515973e-01, -8.253517597287203e-01, 0.0,
         6.598203940130665e-01, 2.037197238535744e-02, 7.065318489492993e-01, 1.0]
    ).reshape((4, 4)).T,
]


def _pose_error(a: Affine, b: Affine) -> float:
    """Combined translation + rotation Frobenius error between two poses."""
    pos = np.linalg.norm(a.translation - b.translation)
    rot = np.linalg.norm(np.asarray(a.matrix)[:3, :3] - np.asarray(b.matrix)[:3, :3])
    return pos + rot


def _redundancy_value(q: np.ndarray, parameter: RedundancyParameter) -> float:
    if parameter == RedundancyParameter.Swivel:
        return swivel_angle(q)
    if parameter == RedundancyParameter.Q7:
        return q[6]
    if parameter == RedundancyParameter.Q6:
        return q[5]
    return q[3]  # Q4


def _displaced_seed(q: np.ndarray, offset: float = 0.6) -> np.ndarray:
    """A seed well away from `q` but sharing its joint 7.

    Holding the seed's Q7 (the default) then guarantees that the branch equal to `q`
    is in the solution set, so tests about *distance* are not silently testing
    reachability instead. Clipped to stay inside the limits.
    """
    lower, upper = (np.asarray(b) for b in PANDA_JOINT_LIMITS)
    seed = np.clip(q - offset, lower, upper)
    seed[6] = q[6]
    return seed


def _random_configs(n: int, limits=PANDA_JOINT_LIMITS, margin: float = 0.05) -> list[np.ndarray]:
    lower, upper = (np.asarray(limits[0]) + margin, np.asarray(limits[1]) - margin)
    rng = np.random.default_rng(0)
    return list(rng.uniform(lower, upper, size=(n, 7)))


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q, expected", list(zip(CONFIGS, LIBFRANKA_FLANGE_POSES)))
def test_forward_kinematics_matches_libfranka(q, expected):
    """The analytical FK must agree with libfranka's model (see LIBFRANKA_FLANGE_POSES)."""
    pose = np.asarray(forward_kinematics(q).matrix)
    np.testing.assert_allclose(pose[:3, 3], expected[:3, 3], atol=1e-9)
    np.testing.assert_allclose(pose[:3, :3], expected[:3, :3], atol=1e-7)


# ---------------------------------------------------------------------------
# Inverse kinematics: branch validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", CONFIGS)
@pytest.mark.parametrize("parameter", ANALYTICAL_PARAMS)
def test_ik_recovers_seed_configuration(q, parameter):
    """The analytical parameterizations should recover the seed config.

    atol is 1e-5 rather than machine-level because accuracy degrades near
    singularities: CONFIGS[2] is the in-plane posture (q1 = q3 = q5 = q7 = 0) and
    Q6 recovers it only to ~1e-6. Away from singularities the error is ~1e-8 or
    better; the pose itself is always reproduced to ~1e-13 (see the tests below).
    """
    target = forward_kinematics(q)  # flange pose
    value = _redundancy_value(q, parameter)
    solutions = inverse_kinematics(target, value, parameter)

    assert solutions, "expected at least one solution"
    # One of the branches must match the original configuration.
    assert any(np.allclose(sol, q, atol=1e-5) for sol in solutions)


@pytest.mark.parametrize("parameter", ALL_PARAMS)
def test_every_returned_branch_reaches_the_target(parameter):
    """Solutions that miss the target pose must be filtered out, not returned.

    Near singularities the solver falls back to branches that do not reach the
    requested pose; IKOptions.pose_tolerance exists to drop them. Randomized over
    the workspace because the bad branches are rare (~0.2% for Q7, ~1.4% for Swivel).
    """
    for q in _random_configs(120):
        target = forward_kinematics(q)
        for sol in inverse_kinematics(target, _redundancy_value(q, parameter), parameter):
            assert _pose_error(forward_kinematics(sol), target) < 1e-5


def test_pose_tolerance_zero_disables_the_check():
    """Disabling the check can only ever admit more branches, never fewer."""
    strict = loose = 0
    for q in _random_configs(120):
        target = forward_kinematics(q)
        options = IKOptions()
        options.pose_tolerance = 0.0
        strict += len(inverse_kinematics(target, q[6], RedundancyParameter.Q7))
        loose += len(inverse_kinematics(target, q[6], RedundancyParameter.Q7, options=options))
    assert loose >= strict


@pytest.mark.parametrize("q", CONFIGS)
def test_solutions_respect_joint_limits(q):
    """No returned branch may violate the configured joint limits."""
    lower, upper = (np.asarray(b) for b in PANDA_JOINT_LIMITS)
    target = forward_kinematics(q)
    for sol in inverse_kinematics(target, q[6], RedundancyParameter.Q7):
        assert np.all(sol >= lower) and np.all(sol <= upper)


def test_unreachable_pose_returns_empty():
    """A target far outside the workspace yields no solutions."""
    target = Affine([10.0, 0.0, 0.0])
    assert inverse_kinematics(target, 0.0) == []
    assert inverse_kinematics_nearest(target, CONFIGS[0]) is None


def test_f_t_ee_is_respected():
    """A non-identity flange->EE transform must round-trip through IK."""
    q = CONFIGS[0]
    f_t_ee = Affine([0.0, 0.0, 0.1034], [0.0, 0.0, -0.3826834, 0.9238795])  # z+0.1034, Rz(-45deg)
    target = forward_kinematics(q, f_t_ee)
    sol = inverse_kinematics_nearest(target, q, f_t_ee=f_t_ee)
    assert sol is not None
    assert _pose_error(forward_kinematics(sol, f_t_ee), target) < 1e-6


# ---------------------------------------------------------------------------
# Swivel: resolved by a sampled search, so only approximately held
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", CONFIGS)
def test_swivel_is_held_approximately(q):
    """The sampled swivel search reaches the pose exactly but holds the swivel angle
    only approximately -- ~2e-3 rad median, ~4e-2 rad worst case at the default sample
    count -- which is why atol here is far looser than for the analytical parameters."""
    target = forward_kinematics(q)
    requested = swivel_angle(q)
    solutions = inverse_kinematics(target, requested, RedundancyParameter.Swivel)

    assert solutions
    for sol in solutions:
        assert _pose_error(forward_kinematics(sol), target) < 1e-5
        assert swivel_angle(sol) == pytest.approx(requested, abs=5e-2)


def test_swivel_samples_out_of_range_raises():
    """swivel_samples feeds fixed-size buffers in the vendored solver: out-of-range
    values must be rejected rather than overrunning them."""
    target = forward_kinematics(CONFIGS[0])
    for samples in (0, MIN_SWIVEL_SAMPLES - 1, MAX_SWIVEL_SAMPLES + 1, 100_000):
        options = IKOptions()
        options.swivel_samples = samples
        with pytest.raises(ValueError):
            inverse_kinematics(target, 0.0, RedundancyParameter.Swivel, options=options)

    # The bounds themselves are valid, and are ignored by the analytical parameters.
    for samples in (MIN_SWIVEL_SAMPLES, MAX_SWIVEL_SAMPLES):
        options = IKOptions()
        options.swivel_samples = samples
        inverse_kinematics(target, 0.0, RedundancyParameter.Swivel, options=options)
    options = IKOptions()
    options.swivel_samples = MAX_SWIVEL_SAMPLES + 1
    assert inverse_kinematics(target, CONFIGS[0][6], RedundancyParameter.Q7, options=options)


# ---------------------------------------------------------------------------
# Nearest-solution helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", CONFIGS)
def test_nearest_holds_current_posture(q):
    """Without an explicit redundancy value the seed's own value for `parameter` is
    held. The default parameter is Q7, which is exact, so the nearest branch is the
    seed itself to solver precision."""
    target = forward_kinematics(q)
    sol = inverse_kinematics_nearest(target, q)
    assert sol is not None
    assert np.allclose(sol, q, atol=1e-5)
    assert _pose_error(forward_kinematics(sol), target) < 1e-5


def test_default_parameter_is_q7():
    """The default must be the exact, microsecond-scale parameter, not the sampled
    swivel search: Swivel costs ~100x and tracks no better (see doc/guide)."""
    q = CONFIGS[0]
    target = forward_kinematics(q)
    implicit = inverse_kinematics(target, q[6])
    explicit = inverse_kinematics(target, q[6], RedundancyParameter.Q7)
    assert len(implicit) == len(explicit)
    assert all(np.allclose(a, b) for a, b in zip(implicit, explicit))
    # Exactness is the observable difference: holding the seed's Q7 recovers the seed,
    # holding its swivel only approximates it.
    assert np.allclose(inverse_kinematics_nearest(target, q), q, atol=1e-5)


@pytest.mark.parametrize("q", CONFIGS)
def test_max_distance_rejects_far_branches(q):
    """max_distance vetoes the result when the nearest branch is a big move, so a
    servo loop gets None instead of a branch-switch jump."""
    target = forward_kinematics(q)
    # The seed reaches the target exactly, so the nearest branch is a zero-distance move
    # and any non-negative bound admits it.
    assert inverse_kinematics_nearest(target, q, max_distance=1e-3) is not None

    # Displace the seed while keeping its joint 7, so the exact branch is guaranteed to
    # still be in the solution set and only the distance to it changes.
    far_seed = _displaced_seed(q)
    unbounded = inverse_kinematics_nearest(target, far_seed)
    assert unbounded is not None
    distance = np.max(np.abs(unbounded - far_seed))
    assert distance > 1e-3, "expected a seed that is genuinely far from every branch"

    # Too tight: rejected outright, rather than answered with some farther branch.
    assert inverse_kinematics_nearest(target, far_seed, max_distance=distance * 0.5) is None
    # Loose enough: the same nearest branch as before, unchanged by the guard.
    bounded = inverse_kinematics_nearest(target, far_seed, max_distance=distance * 1.5)
    assert bounded is not None and np.allclose(bounded, unbounded)
    # The bound is inclusive at the boundary.
    assert inverse_kinematics_nearest(target, far_seed, max_distance=distance) is not None


def test_max_distance_none_accepts_any_move():
    """The guard is opt-in: the default must not silently start rejecting solutions."""
    checked = 0
    for q in _random_configs(60):
        target = forward_kinematics(q)
        far_seed = _displaced_seed(q)
        default = inverse_kinematics_nearest(target, far_seed)
        explicit_none = inverse_kinematics_nearest(target, far_seed, max_distance=None)
        checked += default is not None
        assert (default is None) == (explicit_none is None)
        if default is not None:
            assert np.allclose(default, explicit_none)
    assert checked > 10, "expected the sweep to produce solutions to compare"


def test_nearest_is_the_largest_joint_move_minimizer():
    """`nearest` must minimize the plain L-infinity joint distance over the branches.

    Not a wrapped distance: no Franka joint spans 2*pi, so q_i = +2.8 and q_i = -2.8
    are 5.6 rad apart in the only sense the hardware cares about.
    """
    checked = 0
    for q in _random_configs(60):
        target = forward_kinematics(q)
        seed = q + 0.02
        branches = inverse_kinematics(target, q[6], RedundancyParameter.Q7)
        if len(branches) < 2:
            continue
        checked += 1
        expected = min(branches, key=lambda sol: np.max(np.abs(sol - seed)))
        actual = inverse_kinematics_nearest(
            target, seed, redundancy_value=q[6], parameter=RedundancyParameter.Q7
        )
        assert actual is not None
        assert np.max(np.abs(actual - seed)) == pytest.approx(np.max(np.abs(expected - seed)))
    assert checked > 10, "expected multi-branch cases to exercise the tie-break"


# ---------------------------------------------------------------------------
# Options and constants
# ---------------------------------------------------------------------------


def test_fr3_limits_option_runs():
    """The FR3 limit option should be accepted and produce reproducing solutions."""
    q = CONFIGS[0]
    target = forward_kinematics(q)
    options = IKOptions()
    options.joint_limits = FR3_JOINT_LIMITS
    solutions = inverse_kinematics(target, q[6], RedundancyParameter.Q7, options=options)
    assert solutions
    for sol in solutions:
        assert _pose_error(forward_kinematics(sol), target) < 1e-6


def test_joint_limit_constants_shape():
    for limits in (PANDA_JOINT_LIMITS, FR3_JOINT_LIMITS):
        lower, upper = limits
        assert np.asarray(lower).shape == (7,)
        assert np.asarray(upper).shape == (7,)
        assert np.all(np.asarray(lower) < np.asarray(upper))
        # No joint spans 2*pi; inverse_kinematics_nearest relies on this.
        assert np.all(np.asarray(upper) - np.asarray(lower) < 2 * np.pi)
