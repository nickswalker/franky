#include "franky/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "franky/util.hpp"
#include "geofik.h"

namespace franky {

static_assert(
    kMaxSwivelSamples == GEOFIK_MAX_N_POINTS,
    "franky::kMaxSwivelSamples must match the vendored solver's buffer size");

// The vendored GeoFIK is made reentrant by declaring its mutable globals (scratch
// matrices and the joint-limit configuration) thread_local, so no locking is needed
// here: each thread carries its own scratch and its own limits, which
// inverseKinematics sets before every solve. See third_party/geofik/README.md.

// Constant rigid transform from the Franka flange to GeoFIK's internal 'E' frame
// (the nominal Franka Hand TCP: Rz(-pi/4) then +0.1034 m along z). Computed once
// from GeoFIK itself so we never hard-code the convention here. The static
// initialization is thread-safe (guaranteed by the C++ standard), and franka_fk
// touches no mutable globals, so this is safe to evaluate from any thread.
static const Affine &flangeToGeofikEE() {
  static const Affine transform = []() {
    std::array<double, 7> zero{};
    zero.fill(0.0);
    const Eigen::Matrix4d o_t_flange = franka_fk(zero, 'F');
    const Eigen::Matrix4d o_t_e = franka_fk(zero, 'E');
    return Affine{Eigen::Matrix4d{o_t_flange.inverse() * o_t_e}};
  }();
  return transform;
}

// L-infinity distance between two configurations: the largest single-joint move.
//
// Deliberately not measured modulo 2*pi. No Franka joint has a range wider than 2*pi,
// so a joint angle inside the limits has exactly one valid representative and the
// physical move is the plain difference.
static double distanceLInf(const Vector7d &a, const Vector7d &b) { return (a - b).cwiseAbs().maxCoeff(); }

// Whether q reaches the target pose. The solver's near-singular fallback branches can
// return configurations that miss the target by a millimetre or more, so candidates are
// verified before being handed out. The rotation error is taken from the error
// quaternion, which stays accurate near zero (unlike acos of the trace).
static bool reachesPose(const Vector7d &q, const Affine &target, const Affine &F_T_EE, double tolerance) {
  const Affine achieved = forwardKinematics(q, F_T_EE);
  if ((achieved.translation() - target.translation()).norm() > tolerance) return false;
  const Eigen::Quaterniond error =
      Eigen::Quaterniond(achieved.rotation()).conjugate() * Eigen::Quaterniond(target.rotation());
  const double angle = 2.0 * std::atan2(error.vec().norm(), std::abs(error.w()));
  return angle <= tolerance;
}

static double seedRedundancyValue(const Vector7d &q_seed, RedundancyParameter parameter) {
  switch (parameter) {
    case RedundancyParameter::kSwivel:
      return swivelAngle(q_seed);
    case RedundancyParameter::kQ7:
      return q_seed[6];
    case RedundancyParameter::kQ6:
      return q_seed[5];
    case RedundancyParameter::kQ4:
      return q_seed[3];
  }
  return 0.0;
}

// clang-format off
const JointLimits kPandaJointLimits{
    (Vector7d() << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973).finished(),
    (Vector7d() <<  2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973).finished()};

const JointLimits kFr3JointLimits{
    (Vector7d() << -2.7437, -1.7837, -2.9007, -3.0421, -2.8065,  0.5445, -3.0159).finished(),
    (Vector7d() <<  2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159).finished()};
// clang-format on

Affine forwardKinematics(const Vector7d &q, const Affine &F_T_EE) {
  const Eigen::Matrix4d o_t_flange = franka_fk(toStdD<7>(q), 'F');
  return Affine{o_t_flange} * F_T_EE;
}

double swivelAngle(const Vector7d &q) { return franka_swivel(toStdD<7>(q)); }

std::vector<Vector7d> inverseKinematics(
    const Affine &O_T_EE, double redundancy_value, RedundancyParameter parameter, const Affine &F_T_EE,
    const IKOptions &options) {
  if (parameter == RedundancyParameter::kSwivel &&
      (options.swivel_samples < kMinSwivelSamples || options.swivel_samples > kMaxSwivelSamples)) {
    throw std::invalid_argument(
        "IKOptions::swivel_samples must be in [" + std::to_string(kMinSwivelSamples) + ", " +
        std::to_string(kMaxSwivelSamples) + "], got " + std::to_string(options.swivel_samples));
  }

  // Reconcile franky's end-effector pose with GeoFIK's fixed 'E' frame:
  //   O_T_geofikE = O_T_EE * F_T_EE^-1 * F_T_geofikE.
  const Affine o_t_geofik_ee = O_T_EE * F_T_EE.inverse() * flangeToGeofikEE();

  const Eigen::Vector3d position = o_t_geofik_ee.translation();
  const Eigen::Matrix3d rotation = o_t_geofik_ee.rotation();
  const std::array<double, 3> r = {position.x(), position.y(), position.z()};
  // GeoFIK expects the rotation matrix in row-first (row-major) order.
  const std::array<double, 9> roe = {
      rotation(0, 0), rotation(0, 1), rotation(0, 2),
      rotation(1, 0), rotation(1, 1), rotation(1, 2),
      rotation(2, 0), rotation(2, 1), rotation(2, 2)};

  std::array<std::array<double, 7>, 8> qsols;
  for (auto &sol : qsols) sol.fill(std::numeric_limits<double>::quiet_NaN());

  // GeoFIK's globals are thread_local, so setting the limits and solving affects
  // only this thread; no locking required.
  geofik_set_joint_limits(toStdD<7>(options.joint_limits.first), toStdD<7>(options.joint_limits.second));
  switch (parameter) {
    case RedundancyParameter::kSwivel:
      franka_ik_swivel(r, roe, redundancy_value, qsols, options.q1_singularity, options.swivel_samples);
      break;
    case RedundancyParameter::kQ7:
      franka_ik_q7(r, roe, redundancy_value, qsols, options.q1_singularity);
      break;
    case RedundancyParameter::kQ6:
      franka_ik_q6(r, roe, redundancy_value, qsols, options.q1_singularity, options.q7_singularity);
      break;
    case RedundancyParameter::kQ4:
      franka_ik_q4(r, roe, redundancy_value, qsols, options.q1_singularity, options.q7_singularity);
      break;
  }

  std::vector<Vector7d> solutions;
  solutions.reserve(8);
  for (const auto &sol : qsols) {
    const Vector7d q = toEigenD<7>(sol);
    if (!q.allFinite()) continue;
    if (options.pose_tolerance > 0.0 && !reachesPose(q, O_T_EE, F_T_EE, options.pose_tolerance)) continue;
    solutions.push_back(q);
  }
  return solutions;
}

std::optional<Vector7d> inverseKinematicsNearest(
    const Affine &O_T_EE, const Vector7d &q_seed, const Affine &F_T_EE, const IKOptions &options,
    std::optional<double> redundancy_value, RedundancyParameter parameter,
    std::optional<double> max_distance) {
  const double value = redundancy_value.value_or(seedRedundancyValue(q_seed, parameter));
  const std::vector<Vector7d> solutions = inverseKinematics(O_T_EE, value, parameter, F_T_EE, options);

  std::optional<Vector7d> best;
  double best_distance = std::numeric_limits<double>::infinity();
  for (const Vector7d &solution : solutions) {
    const double distance = distanceLInf(solution, q_seed);
    if (distance < best_distance) {
      best_distance = distance;
      best = solution;
    }
  }
  // Vetoed after the search, not during it, so the result is always the nearest branch
  // and never a farther one that happens to fit under the bound.
  if (max_distance.has_value() && best_distance > *max_distance) return std::nullopt;
  return best;
}

}  // namespace franky
