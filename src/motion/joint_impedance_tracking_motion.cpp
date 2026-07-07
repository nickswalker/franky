#include "franky/motion/joint_impedance_tracking_motion.hpp"

namespace franky {

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(std::shared_ptr<JointReferenceHandle> reference_handle)
    : JointImpedanceTrackingMotion(std::move(reference_handle), Params{}) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(
    std::shared_ptr<JointReferenceHandle> reference_handle, const Params &params)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params), reference_handle_(std::move(reference_handle)) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(
    std::shared_ptr<JointReferenceHandle> reference_handle, const Params &params, RuntimeOptions runtime)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params, std::move(runtime)),
      reference_handle_(std::move(reference_handle)) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(ReferenceCallback reference_callback)
    : JointImpedanceTrackingMotion(std::move(reference_callback), Params{}) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params),
      reference_callback_(std::move(reference_callback)) {}

void JointImpedanceTrackingMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  JointReference reference;
  reference.q = robot_state.q;
  applied_reference_.set(reference);
}

franka::Torques JointImpedanceTrackingMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  JointReference reference;
  reference.q = target_;
  reference.dq = target_velocity_;

  if (reference_callback_) {
    reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
  } else if (reference_handle_ && reference_handle_->hasReference()) {
    reference = reference_handle_->get();
  }

  applied_reference_.set(reference);
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

}  // namespace franky
