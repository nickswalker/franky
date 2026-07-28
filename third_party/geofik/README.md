# GeoFIK (vendored)

Analytical geometric inverse kinematics for the Franka arm.

- **Upstream:** https://github.com/PabloLopezCustodio/GeoFIK
- **Author:** Pablo Lopez-Custodio
- **License:** MIT (see `LICENSE`). Relicensed to MIT for franky with the author's
  explicit permission; the upstream repository did not carry a license file at the
  time of vendoring.
- **Paper:** Lopez-Custodio et al., "GeoFIK: A Fast Geometric Solver for the
  Inverse Kinematics of the Franka Robot."

franky consumes this through the wrapper in `franky/kinematics.hpp` /
`src/kinematics.cpp`. Application code should use that wrapper, not these headers
directly.

## Local modifications

The vendored files are kept as close to upstream as possible to ease future
re-syncing. The only changes:

`geofik.h`
- Removed the global `using namespace std;` (it leaked `std` into every includer).
  Declarations are now qualified with `std::`. The `.cpp` keeps a file-local
  `using namespace std;` so the implementation is untouched.
- `#include <eigen3/Eigen/Dense>` → `#include <Eigen/Dense>` to match franky's
  Eigen include convention (`Eigen3::Eigen`).
- Renamed the global `PI` constant guard to `GEOFIK_PI` (kept `PI` as an alias) and
  added the `geofik_set_joint_limits(...)` declaration.
- Exposed `GEOFIK_MAX_N_POINTS` (the size of the swivel solver's sweep buffers, 1000).
  Upstream keeps that bound private to the `.cpp` and does not check the caller's
  `n_points` against it, so a larger `n_points` writes past the buffers. The franky
  wrapper now rejects out-of-range `IKOptions::swivel_samples` using this constant.

`geofik.cpp`
- Added `#include <cmath>` and a file-local `using namespace std;`.
- Made `q_low` / `q_up` / `q_mid` mutable (were `const`) and added
  `geofik_set_joint_limits(lower, upper)`, which also recomputes `q_mid` as the
  midpoint of the limits. Defaults are unchanged (Panda), so behavior matches
  upstream unless the setter is called. This is how the wrapper switches to FR3
  limits.
- Declared the nine mutable file-scope globals `thread_local`: the six solver
  scratch variables (`tmp_R`, `tmp_J`, `tmp_J_6d`, `J_old`, `J_old_low`, `s`) and
  the three joint-limit arrays (`q_low`, `q_up`, `q_mid`). See "Thread safety".
- `MAX_N_POINTS` is now defined from the public `GEOFIK_MAX_N_POINTS` (same value).
- Diagnostics no longer go to `std::cout`. Upstream prints messages such as
  `ERROR: unable to assembly kinematic chain` on ordinary inputs — an unreachable
  target or a singular pose — which for a control library means stray writes to
  stdout during normal operation (the outcome is already reported by returning zero
  solutions). The 14 `cout <<` statements now write to `geofik_log()`, a null sink,
  or to `std::cerr` when `GEOFIK_VERBOSE` is defined at compile time.

## Thread safety

Upstream GeoFIK uses file-scope **mutable** scratch globals (`tmp_R`, `J_old`,
`s`, …), which makes it non-reentrant. Rather than serialize every call behind a
mutex, franky declares those globals (and the mutable joint-limit globals) as
`thread_local`. Each thread then operates on its own copies, so concurrent IK
calls do not race and no locking is required.

Consequences of the `thread_local` approach:
- `geofik_set_joint_limits` sets the limits for the **calling thread only**. The
  franky wrapper calls it before every solve, so this is transparent to users.
- Read-only globals (`J0_S`, `MAX_N_POINTS`, the `#define`d DH constants) are left
  shared — they are never written, so sharing them is safe.
- The per-thread cost is a handful of small Eigen matrices and TLS-local access,
  negligible next to the solver's trigonometry.
