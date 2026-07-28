# Analytical Inverse Kinematics

franky ships an analytical inverse kinematics (IK) solver for the Franka arm, backed by the
[GeoFIK](https://github.com/PabloLopezCustodio/GeoFIK) geometric solver (vendored under
`third_party/geofik`, MIT licensed). Unlike commanding a `CartesianMotion` — where the robot's
controller resolves the pose to joint angles internally — these functions compute joint
configurations **on the client side, without a robot connection**. That makes them useful for
reachability checks, planning, seeding motions, converting a Cartesian goal into a `JointMotion`,
and offline or simulation work.

The solver uses the nominal Franka DH parameters, so it does not account for per-robot factory
calibration.

## Forward kinematics

`forward_kinematics` computes the end-effector pose for a joint configuration. With no
flange-to-end-effector transform it returns the flange pose.

```python
import numpy as np
from franky import forward_kinematics, Affine

q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
flange_pose: Affine = forward_kinematics(q)
```

## Resolving the redundancy

The Franka arm is redundant: a given end-effector pose is reachable by a one-parameter family of
configurations. You pick one by fixing a `RedundancyParameter`:

- `Q7` (default), `Q6`, `Q4`: hold the respective joint angle. Fully analytical, exact, and a few
  microseconds per call.
- `Swivel`: the swivel (arm) angle — the rotation of the elbow about the shoulder-wrist axis. This
  is the intuitive "elbow" coordinate, so reach for it when the elbow posture is the thing you want
  to specify. It is resolved by a sampled search rather than in closed form, so it costs roughly two
  orders of magnitude more and holds the requested angle only approximately (median ~2e-3 rad, ~1e-2
  rad at the 95th percentile, worst case ~4e-2 rad).

`Swivel` was the default in earlier versions of this branch. It is not any more: for following a
Cartesian path it buys nothing over `Q7` (see the table below), and holding the seed's joint 7 is
the standard Franka convention.

Whichever you pick, every returned configuration reaches the requested pose: candidates that miss it
(the solver has near-singular fallback branches that do) are dropped, controlled by
`IKOptions.pose_tolerance`.

`inverse_kinematics` returns **all** valid branches (up to 8), dropping any that violate the joint
limits:

```python
from franky import inverse_kinematics, RedundancyParameter, swivel_angle

target = forward_kinematics(q)
# All configurations that reach `target` with the same joint 7 as `q`:
solutions = inverse_kinematics(target, q[6])
# ... or with the same elbow posture as `q`, at ~100x the cost:
solutions = inverse_kinematics(target, swivel_angle(q), RedundancyParameter.Swivel)
```

To reach a pose while staying close to a known configuration (e.g. the current one), use
`inverse_kinematics_nearest`. With no explicit redundancy value it holds the seed's own value, i.e.
keeps the current arm posture:

```python
from franky import inverse_kinematics_nearest

q_solution = inverse_kinematics_nearest(target, q_seed=q)  # nearest branch, current posture held
```

```c++
#include <franky.hpp>
using namespace franky;

Vector7d q;
q << 0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8;
Affine target = forwardKinematics(q);
std::vector<Vector7d> solutions = inverseKinematics(target, q[6]);
std::optional<Vector7d> nearest = inverseKinematicsNearest(target, q);
```

"Nearest" means nearest among the branches that reach the pose within the joint limits, measured as
the largest single-joint move. That set changes along a path, so successive calls can switch branches
and hand back a configuration several radians from the seed. **No choice of redundancy parameter
avoids this** — it is a property of picking from a discrete branch set, and the worst jumps measured
were as large under `Swivel` as under `Q7`.

Pass `max_distance` to convert such a jump into an explicit `None`:

```python
q_solution = inverse_kinematics_nearest(target, q_seed=q, max_distance=0.1)
if q_solution is None:
    ...  # unreachable, or reachable only by a branch switch: hold, replan, or slow down
```

The guard is a veto applied after the search, so what you get back is always the nearest branch —
never a farther one that happened to fit under the bound. Be ready for `None` even without it: the
redundancy value is held at the seed's, which is not always reachable at the next pose.

## Accuracy and cost

Measured against libfranka 0.21 on 2000 random configurations (Release build, x86-64), with the
target pose taken as the forward kinematics of a legal configuration:

| | output | wallclock |
|---|---|---|
| `forward_kinematics` vs `franka::Model::pose` | agree to 1.4e-13 m / 5.2e-8 rad | 0.6 us vs 7.8 us |
| `inverse_kinematics`, `Q7` / `Q6` / `Q4` | recovers the source configuration on 100% of targets; pose reproduced to ~1e-13 | 5-9 us for all branches (3-5 of them) |
| `inverse_kinematics`, `Swivel` | reaches the pose, holds the swivel angle to ~2e-3 rad median / ~4e-2 rad worst case; no solution on ~3% of reachable targets | 250-500 us |
| DLS iteration on `franka::Model::zero_jacobian` (for reference) | converges from a nearby seed; ~57% failure from a random seed | 40-70 us near the seed, ~1.5 ms far from it |

The analytical solver's advantage over iterating on libfranka's Jacobian is that it enumerates all
branches and does not depend on a seed. Note that it uses the nominal DH parameters, so unlike the
robot's own controller it ignores per-robot factory calibration — expect sub-millimetre disagreement
with where the robot actually goes.

`IKOptions.swivel_samples` trades cost against coverage roughly linearly (2000 random targets):

| samples | us/call | targets solved | swivel residual (median / p95 / max) |
|---|---|---|---|
| 30 | 22 | 68.5% | 1.3e-2 / 4.0e-2 / 4.5e-2 |
| 100 | 55 | 92.9% | 7.7e-3 / 3.6e-2 / 4.5e-2 |
| 300 | 114 | 98.3% | 3.5e-3 / 1.9e-2 / 4.4e-2 |
| 600 (default) | 211 | 99.2% | 2.3e-3 / 1.0e-2 / 4.3e-2 |
| 1000 (max) | 352 | 99.5% | 1.6e-3 / 7.3e-3 / 4.1e-2 |

Read it as a coverage knob rather than an accuracy knob. Lowering it mainly costs you targets the
sweep fails to bracket at all, which is why the default is not lower. The median residual does shrink
as roughly 1/samples, but the worst case is flat: the solver refines its bracket only by local
interpolation.

### Following a path

Measured over 60 straight-line 50 mm paths in 2 mm steps, every step independently verified reachable,
seeding each call with the previous solution and stopping the path at the first failure. "Jump" is the
largest single-joint move between consecutive steps:

| parameter | us/step | steps tracked | jump median / p99 / max |
|---|---|---|---|
| `Swivel` | 304 | 99.8% | 0.010 / 0.35 / 4.90 |
| `Q7` | 3.2 | 99.2% | 0.009 / 0.38 / 2.42 |
| `Q6` | 3.0 | 99.3% | 0.010 / 0.18 / 3.06 |
| `Q4` | 5.5 | 95.9% | 0.031 / 1.19 / 5.48 |

`Swivel` costs ~100x for 0.6 points of tracking rate and no better continuity — hence the `Q7`
default. The multi-radian tails are branch switches, present under every parameter, which is what
`max_distance` is for. `Q4` is the one to avoid for tracking.

## Using a robot's current state

When you have a `Robot`, the convenience methods fill in the current flange-to-end-effector
transform (and, for the nearest variant, seed with the current joint positions):

```python
q_solution = robot.inverse_kinematics_nearest(target)  # reach `target`, hold current posture
q_solution = robot.inverse_kinematics_nearest(target, max_distance=0.1)  # ... or None on a big move
```

## Panda vs. FR3 joint limits

Solutions are filtered against joint limits, defaulting to the Panda's. For an FR3, pass the FR3
limits via `IKOptions`:

```python
from franky import IKOptions, FR3_JOINT_LIMITS

options = IKOptions()
options.joint_limits = FR3_JOINT_LIMITS
solutions = inverse_kinematics(target, q[6], options=options)
```

The kinematic geometry is identical between the Panda and the FR3; only the joint limits differ. The
limits also decide which branches survive, so passing the wrong ones silently changes the result set.

## Bridging to `CartesianMotion`

A full IK solution `q` gives you the elbow directly (`q[2]` and `sign(q[3])`), so you can drive a
`CartesianMotion` with a specific elbow configuration, or command the whole configuration with a
`JointMotion`.
```
