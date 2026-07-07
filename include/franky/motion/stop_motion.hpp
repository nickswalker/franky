#pragma once

#include <franka/control_types.h>

#include <optional>

#include "franky/motion/cartesian_motion.hpp"
#include "franky/motion/cartesian_velocity_motion.hpp"
#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/joint_motion.hpp"
#include "franky/motion/joint_velocity_motion.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/reference_type.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/robot_state.hpp"
#include "franky/types.hpp"

namespace franky {

template <typename ControlSignalType>
class StopMotion;

/**
 * @brief Stop motion for joint position control mode.
 */
template <>
class StopMotion<franka::JointPositions> : public JointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointMotion(JointState(Vector7d::Zero()), ReferenceType::kRelative, relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for joint velocity position control mode.
 */
template <>
class StopMotion<franka::JointVelocities> : public JointVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointVelocityMotion(Vector7d::Zero(), franka::Duration(0), relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian pose control mode.
 */
template <>
class StopMotion<franka::CartesianPose> : public CartesianWaypointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianWaypointMotion(
            {PositionWaypoint<CartesianState>{
                {
                    .target = RobotPose(),
                    .hold_target_duration = franka::Duration(50),
                },
                ReferenceType::kRelative}},
            relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian velocity control mode.
 */
template <>
class StopMotion<franka::CartesianVelocities> : public CartesianVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianVelocityMotion(RobotVelocity(), franka::Duration(0), relative_dynamics_factor) {}
};

/**
 * @brief Parameters for the torque-control stop motion (StopMotion<franka::Torques>).
 */
struct TorqueStopParams {
  /** Pure joint damping [Nms/rad] used to bring the arm to rest. */
  Vector7d damping{defaultJointImpedanceDamping()};

  /**
   * Duration [s] over which the inherited command is ramped into the damping-only
   * law, avoiding a torque discontinuity at the handover. Set to 0 to switch
   * immediately (the rate limiter still smooths the transition).
   */
  double ramp_duration{0.2};

  /** Finish once every joint speed is below this [rad/s] (after the ramp completes). */
  double velocity_epsilon{0.02};

  /**
   * Hard cap [s]: finish regardless of velocity so a sustained external push can
   * never make the stop hang forever.
   */
  double max_duration{2.0};

  /** Compensate Coriolis forces, matching the impedance controllers for a seamless ramp. */
  bool compensate_coriolis{true};

  /** Per-tick torque-rate limit [Nm] for smoothness. */
  double max_delta_tau{1.0};
};

/**
 * @brief Graceful stop for torque-control (impedance) motions.
 *
 * Torque motions never signal MotionFinished on their own, and franka::Robot::stop()
 * preempts the control loop with a franka::ControlException rather than ramping down.
 * Enqueue this motion (or return it from a TorqueReaction) to end a torque loop
 * cleanly: it blends the last commanded torque into a zero-stiffness joint-damping law
 * (so it yields to the arm's current pose instead of snapping to a target), brings the
 * arm to rest, then returns franka::MotionFinished.
 */
template <>
class StopMotion<franka::Torques> : public Motion<franka::Torques> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit StopMotion(const TorqueStopParams &params = {}) : params_(params) {}

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  TorqueStopParams params_;
  // Torque we ramp *from* — the inherited command, captured in initImpl.
  Vector7d tau_start_{Vector7d::Zero()};
};

}  // namespace franky
