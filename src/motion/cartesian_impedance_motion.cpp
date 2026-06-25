#include "franky/motion/cartesian_impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>

#include "franky/motion/cartesian_impedance_base.hpp"

namespace franky {

CartesianImpedanceMotion::CartesianImpedanceMotion(const Affine &target, franka::Duration duration)
    : CartesianImpedanceMotion(target, duration, Params()) {}

CartesianImpedanceMotion::CartesianImpedanceMotion(
    const Affine &target, franka::Duration duration, const CartesianImpedanceMotion::Params &params)
    : CartesianImpedanceBase(target, params), duration_(duration), params_(params) {}

void CartesianImpedanceMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  CartesianImpedanceBase::initImpl(robot_state, previous_command);
  if (params_.target_type == ReferenceType::kRelative) setAbsoluteTarget(robot_state.O_T_EE * target_spec());
  initial_pose_ = robot_state.O_T_EE;
}

std::tuple<CartesianReference, bool> CartesianImpedanceMotion::update(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration time, franka::Duration /*abs_time*/) {
  double transition_parameter = duration_.toSec() > 0.0 ? time.toSec() / duration_.toSec() : 1.0;
  // Past the interpolation window the motion holds the target, either indefinitely or until
  // finish_wait_factor expires.
  Affine intermediate_goal = target();
  bool done = false;
  if (transition_parameter <= 1.0) {
    Eigen::Quaterniond q_start(initial_pose_.rotation());
    Eigen::Quaterniond q_end(target().rotation());
    auto init_trans = initial_pose_.translation();
    auto trans = init_trans + transition_parameter * (target().translation() - init_trans);
    auto rot = q_start.slerp(transition_parameter, q_end);
    intermediate_goal.fromPositionOrientationScale(trans, rot, Eigen::Vector3d::Ones());
  } else if (params_.return_when_finished && transition_parameter > params_.finish_wait_factor) {
    done = true;
  }
  return {CartesianReference{intermediate_goal, std::nullopt}, done};
}

}  // namespace franky
