#pragma once

#include <functional>
#include <memory>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/realtime_value.hpp"

namespace franky {

/**
 * @brief Handle for updating a CartesianReference online (see RealtimeValue).
 *
 * This handle is intended to be written from a user thread while a single
 * CartesianImpedanceTrackingMotion is running. The motion reads the latest
 * valid reference each control cycle without needing to replace the motion
 * object.
 *
 * Thread safety: at most one thread may call set() or clear() at a time (the
 * writer), and get() is reserved for the single RT reader. hasReference() is
 * safe from either thread.
 */
class CartesianReferenceHandle {
 public:
  CartesianReferenceHandle() = default;

  void set(const CartesianReference &reference) { value_.set(reference); }
  void clear() { value_.clear(); }
  [[nodiscard]] bool hasReference() const { return value_.hasValue(); }
  //! Latest published reference. RT reader thread only.
  [[nodiscard]] const CartesianReference &get() const { return value_.get(); }

 private:
  RealtimeValue<CartesianReference> value_;
};

/**
 * @brief Cartesian impedance tracking motion.
 *
 * This motion keeps the same Cartesian impedance controller alive while
 * reading the latest reference from a handle or callback every control cycle.
 */
class CartesianImpedanceTrackingMotion : public CartesianImpedanceBase {
 public:
  using Params = CartesianImpedanceBase::Params;
  using ReferenceCallback =
      std::function<CartesianReference(const RobotState &, franka::Duration, franka::Duration, franka::Duration)>;

  explicit CartesianImpedanceTrackingMotion(std::shared_ptr<CartesianReferenceHandle> reference_handle);
  CartesianImpedanceTrackingMotion(std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params);
  CartesianImpedanceTrackingMotion(
      std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params, RuntimeOptions runtime);
  explicit CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback);
  CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params);

  [[nodiscard]] const Params &params() const { return base_params(); }

  // target() intentionally shadows
  // the by-value accessor on CartesianImpedanceBase, which reflects the static absolute_target_ and
  // is meaningless for a tracking motion.
  [[nodiscard]] Affine target() const { return applied_reference_.get().target; }
  [[nodiscard]] std::optional<Twist> target_twist() const { return applied_reference_.get().target_twist; }
  [[nodiscard]] std::optional<TwistAcceleration> target_acceleration() const {
    return applied_reference_.get().target_acceleration;
  }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  std::tuple<CartesianReference, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time,
      franka::Duration abs_time) override;

 private:
  std::shared_ptr<CartesianReferenceHandle> reference_handle_;
  ReferenceCallback reference_callback_;
  Affine target_;
  std::optional<Twist> target_twist_;
  std::optional<TwistAcceleration> target_acceleration_;

  RealtimeValue<CartesianReference> applied_reference_;
};

}  // namespace franky
