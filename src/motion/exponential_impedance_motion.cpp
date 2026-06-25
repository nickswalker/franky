#include "franky/motion/exponential_impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <map>

#include "franky/motion/cartesian_impedance_base.hpp"

namespace franky {

ExponentialImpedanceMotion::ExponentialImpedanceMotion(const Affine &target)
    : ExponentialImpedanceMotion(target, Params()) {}

ExponentialImpedanceMotion::ExponentialImpedanceMotion(
    const Affine &target, const ExponentialImpedanceMotion::Params &params)
    : CartesianImpedanceBase(target, params), params_(params) {}

std::tuple<CartesianReference, bool> ExponentialImpedanceMotion::update(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration time, franka::Duration /*abs_time*/) {
  constexpr double kNominalDt = 1e-3;
  const double alpha = 1.0 - std::pow(1.0 - params_.exponential_decay, time_step.toSec() / kNominalDt);
  auto trans = alpha * target().translation() + (1.0 - alpha) * intermediate_target().translation();
  auto rot = Eigen::Quaterniond(intermediate_target().rotation()).slerp(alpha, Eigen::Quaterniond(target().rotation()));
  return {
      CartesianReference{Affine().fromPositionOrientationScale(trans, rot, Eigen::Vector3d::Ones()), std::nullopt},
      false};
}

}  // namespace franky
