# Manipulability benchmark

Answers a specific question: the Cartesian impedance controller's manipulability nullspace task
derives its gradient from a `JacobiSVD` pseudo-inverse plus seven `franka::Model::pose()` calls for
the joint origins (`src/motion/cartesian_impedance_base.cpp`). Now that an analytical Jacobian and
forward kinematics are vendored in the tree (`third_party/geofik`), is a different formulation
worth it?

## Running

The benchmark is a standalone CMake project. It links Eigen and the vendored GeoFIK only, so it
builds and runs without libfranka, ruckig, or a robot.

```
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build build/benchmarks
./build/benchmarks/bench_manipulability --configs 2048 --reps 20
```

Configurations are sampled uniformly inside the Panda joint limits with a small inset. Each variant
is timed once per configuration with one clock read per call (back-to-back `steady_clock` reads cost
0.02 us here, ~0.3% of the fastest variant).

## What is measured

| Variant | Description |
| --- | --- |
| A | The shipped implementation, ported verbatim: `sqrt(det(J J^T))`, `JacobiSVD` pseudo-inverse, analytic `dJ/dq` |
| B | Same math, but the pseudo-inverse and `w` both come out of one Cholesky of `J J^T` |
| C | B with `dJ` never materialized — the trace is accumulated straight out of the cross products |
| D | No analytic `dJ` at all: central differences of `w(q)` using the analytical Jacobian (14 evaluations) |
| E | Householder QR of `J^T`, so `J J^T` is never formed; `w = abs(prod(diag(R)))`, `(J^#)^T = R^-1 Q^T` |

The identity that makes B/C/E possible: for full-row-rank `J`, `J^# = J^T (J J^T)^-1`, so
`(J^#)^T = (J J^T)^-1 J` — which is exactly the matrix the trace `tr(J^# dJ)` contracts against, and
also exactly `pinv(J^T)`, the matrix `computeCommand` builds the nullspace projector from with a
*second* SVD. One factorization can serve `w`, the manipulability pseudo-inverse, and the projector.

## Caveat on the numbers

`franka::Model::pose()` cannot be timed offline — `franka::Model` needs the model library libfranka
downloads from the robot. Every variant therefore sources its kinematics from GeoFIK. That makes the
comparison *between variants* exact, and makes the reported cost of variant A a **lower bound** on
what the shipped code costs: a cross-shared-object call into libfranka's generated model code cannot
be cheaper than the inlined GeoFIK sweep standing in for it.

The measurements below were taken in a shared cloud container, not on a PREEMPT_RT host. Medians are
reproducible to within a few percent across runs; the `max` column is dominated by scheduler
preemption (occasional ~1 ms outliers) and says nothing about the algorithms. Read the medians and
the ratios.

## Results

2048 configurations, 20 repetitions, x86-64, GCC 13.3, `-O2`.

### 1. Gradient algebra alone (Jacobian and joint origins supplied)

```
  variant                                     med (us)  p99 (us)  vs base
  A. current (JacobiSVD pinv + dJ)               7.25     22.72
  B. LLT pinv + dJ                               0.98      1.61     7.5x
  C. LLT pinv + fused dJ (no dJ matrix)          0.87      1.37     8.7x
  D. central differences via analytic J (14x)   12.89     32.41     0.6x
  E. QR of J^T + fused dJ (no J J^T formed)      1.66      4.72     4.4x
```

### 2. Kinematics sourcing

```
  joint origins, 7 independent FK calls          4.73      9.19
  joint origins, 1 forward sweep                 0.74      1.14     6.4x
  analytic Jacobian (J_from_q)                   0.79      1.46     5.9x
```

Seven independent FK calls redo the chain from the base each time. `franka_fk_all_frames` gets every
joint origin in one sweep.

### 3. Gradient including its own kinematics

```
  A. current shape (7 FK + SVD + dJ)            12.14     31.96
  C. 1 sweep + LLT + fused dJ                    1.63      2.52     7.5x
  D. central differences (no kinematics needed) 12.89     31.50     0.9x
```

### 4. Full nullspace block (projector + manipulability torque)

This is the block `computeCommand` actually runs whenever any nullspace task is active, and it is
where the second `JacobiSVD` lives.

```
  A. current (2 JacobiSVDs)                     13.74     34.43
  B. one LLT shared by projector + gradient      0.96      1.49    14.3x
  C. one QR shared by projector + gradient       3.01      6.23     4.6x
```

### 5. Accuracy on well-conditioned configurations

Relative error against a Richardson-extrapolated central-difference reference, `O(h^4)`:

```
  A. current (JacobiSVD pinv + dJ)          med 7.06e-11   max 3.12e-08
  C. LLT pinv + fused dJ                    med 7.06e-11   max 3.12e-08
  D. central differences, h=1e-5            med 2.95e-10   max 2.37e-07
  E. QR pinv + fused dJ                     med 7.06e-11   max 3.12e-08
```

All the analytic variants are the same function to the last bit that matters. Away from
singularities this is purely a speed question.

### 6. Near-singular behaviour

Descent on `w` until the arm is genuinely rank-deficient, scored against `w = prod(sigma_i(J))` taken
from the singular values of `J` itself (which never forms `J J^T`, so it does not square the
condition number):

```
  w (sigmas)     det rel    chol rel      qr rel     llt ok
  8.67e-02      3.52e-15    2.72e-15    3.04e-15       yes
  8.59e-03      3.84e-15    2.02e-15    2.02e-15       yes
  7.70e-04      2.47e-13    5.24e-13    4.22e-16       yes
  4.46e-05      5.99e-11    1.07e-10    5.96e-14       yes
  4.43e-06      4.18e-09    3.26e-09    7.41e-13       yes
  4.36e-07      2.07e-07    1.62e-07    3.48e-12       yes
  3.03e-08      1.49e-04    1.93e-04    4.92e-11       yes
  8.04e-10      5.11e-02    3.42e-02    1.34e-09       yes
  2.34e-11      1.00e+00    1.00e+00    9.43e-09        NO
```

This is the result that is easy to get wrong. Switching `det(J J^T)` for a Cholesky of `J J^T` buys
**nothing** numerically — the damage is done by forming `J J^T` at all, which squares the condition
number, and it does not matter how the product is subsequently factored. Both routes have lost all
significant digits in `w` by `w ~ 1e-11`, and the Cholesky additionally fails outright once the
product goes numerically indefinite (variants B and C return a zero gradient there; the shipped code
degrades more gracefully via its singular-value tolerance).

Factoring `J^T` directly with a Householder QR avoids the squaring and holds ~9 correct digits all
the way down.

## Reading

- The speed win is real and large, but **it has almost nothing to do with the analytical IK**. Two
  thirds of it is dropping `JacobiSVD` — the shipped code runs two full SVDs with `ComputeFullU |
  ComputeFullV` per tick whenever a nullspace task is active, and both of them are computing a
  matrix that one 6x6 factorization already provides. That change is available without touching
  GeoFIK at all.
- Where GeoFIK *does* help is the joint-origin lookup: 4.7 us of seven separate FK calls collapses to
  0.7 us in one sweep, and the manipulability task stops needing `franka::Model` — i.e. stops needing
  a robot connection — which makes it testable offline.
- Finite differences on the analytical Jacobian, the most obvious "now that we have it" rewrite, is
  the one option the benchmark rules out: 12.9 us, slower than what is shipped today, less accurate,
  and it throws away a correct analytic derivative to get there.
- On absolute terms, 13.7 us against a 1 ms tick is 1.4% of the budget, so nothing is on fire. The
  argument for changing it is the p99 (34 us) and the fact that the same factorization is being paid
  for twice, not a missed deadline.
- If the change is made, prefer the QR (variant E). It is ~2 us slower than the Cholesky in the full
  block and still 4.6x faster than what is shipped, it never fails, and it is the only variant that
  stays accurate in exactly the region a manipulability-maximizing controller is designed to operate
  in. Trading seven orders of magnitude of near-singular accuracy for 2 us is a bad trade here.
