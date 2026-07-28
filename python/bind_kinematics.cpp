#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <utility>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_kinematics(py::module &m) {
  py::enum_<RedundancyParameter>(
      m, "RedundancyParameter", "Free variable used to resolve the Franka arm's 1-DOF redundancy.")
      .value(
          "Swivel",
          RedundancyParameter::kSwivel,
          "Swivel (arm) angle: rotation of the elbow about the shoulder-wrist axis. The intuitive "
          "\"elbow\" coordinate; use it when the elbow posture is what you want to specify. Resolved "
          "by a sampled search, so it is far slower than the analytical parameters, holds the "
          "requested angle only approximately, and offers no advantage for path tracking.")
      .value(
          "Q7",
          RedundancyParameter::kQ7,
          "Angle of joint 7 (wrist). Fully analytical, exact, and the default.")
      .value("Q6", RedundancyParameter::kQ6, "Angle of joint 6. Fully analytical.")
      .value("Q4", RedundancyParameter::kQ4, "Angle of joint 4 (elbow joint). Fully analytical.");

  py::class_<IKOptions>(m, "IKOptions", "Options for the analytical inverse kinematics.")
      .def(
          py::init<>([](JointLimits joint_limits, double q1_singularity, double q7_singularity,
                        unsigned int swivel_samples, double pose_tolerance) {
            IKOptions options;
            options.joint_limits = std::move(joint_limits);
            options.q1_singularity = q1_singularity;
            options.q7_singularity = q7_singularity;
            options.swivel_samples = swivel_samples;
            options.pose_tolerance = pose_tolerance;
            return options;
          }),
          py::arg_v("joint_limits", kPandaJointLimits, "PANDA_JOINT_LIMITS"),
          "q1_singularity"_a = detail::kPi / 2,
          "q7_singularity"_a = 0.0,
          "swivel_samples"_a = 600,
          "pose_tolerance"_a = 1e-6)
      .def_readwrite(
          "joint_limits",
          &IKOptions::joint_limits,
          "Joint limits (lower, upper) [rad] used to reject solutions. Defaults to PANDA_JOINT_LIMITS; "
          "set to FR3_JOINT_LIMITS for an FR3.")
      .def_readwrite(
          "q1_singularity",
          &IKOptions::q1_singularity,
          "Fallback value of joint 1 in a shoulder (type-1) singularity [rad].")
      .def_readwrite(
          "q7_singularity",
          &IKOptions::q7_singularity,
          "Fallback value of joint 7 in a wrist (type-2) singularity [rad]. Ignored for Q7.")
      .def_readwrite(
          "swivel_samples",
          &IKOptions::swivel_samples,
          "Number of samples used when resolving the redundancy by swivel angle. Ignored otherwise. Must be in "
          "[MIN_SWIVEL_SAMPLES, MAX_SWIVEL_SAMPLES]; other values raise ValueError. Cost is linear in this value. "
          "A coverage knob rather than an accuracy knob: lowering it mainly costs you targets whose solution the "
          "sweep fails to bracket at all. The typical swivel residual shrinks as samples are added; the worst "
          "case does not.")
      .def_readwrite(
          "pose_tolerance",
          &IKOptions::pose_tolerance,
          "Solutions whose forward kinematics misses the target by more than this in position [m] or rotation "
          "angle [rad] are dropped. Guards against the solver's near-singular fallback branches. Set to 0 to "
          "disable the check.");

  m.attr("PANDA_JOINT_LIMITS") = kPandaJointLimits;
  m.attr("FR3_JOINT_LIMITS") = kFr3JointLimits;
  m.attr("MIN_SWIVEL_SAMPLES") = kMinSwivelSamples;
  m.attr("MAX_SWIVEL_SAMPLES") = kMaxSwivelSamples;

  // These hold the GIL: they are short enough that a release/reacquire pair costs about
  // as much as the call itself. The IK entry points below release conditionally.
  m.def(
      "forward_kinematics",
      &forwardKinematics,
      "q"_a,
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "Analytical forward kinematics using the nominal Franka geometry (no robot connection required).\n\n"
      "Args:\n"
      "    q: Joint angles [rad].\n"
      "    f_t_ee: Transform from flange to end-effector frame. Default (identity) returns the flange pose.\n\n"
      "Returns:\n"
      "    The pose of the end-effector frame relative to the base frame.");

  m.def(
      "swivel_angle",
      &swivelAngle,
      "q"_a,
      "The swivel (arm) angle of a configuration [rad]. Inverse of resolving the redundancy by "
      "RedundancyParameter.Swivel; use it to hold the current arm posture across an IK call.");

  // A Swivel solve is long enough that holding the GIL across it can push a Python thread
  // running a control or state-polling loop past its deadline; the analytical parameters
  // are short enough that releasing costs more than it saves. py::call_guard would be
  // all-or-nothing, hence the conditional release. Safe here: the solver invokes no
  // Python, and GeoFIK's globals are thread_local (see third_party/geofik/README.md).
  m.def(
      "inverse_kinematics",
      [](const Affine &o_t_ee, double redundancy_value, RedundancyParameter parameter, const Affine &f_t_ee,
         const IKOptions &options) {
        std::optional<py::gil_scoped_release> release;
        if (parameter == RedundancyParameter::kSwivel) release.emplace();
        return inverseKinematics(o_t_ee, redundancy_value, parameter, f_t_ee, options);
      },
      "o_t_ee"_a,
      "redundancy_value"_a,
      py::arg_v("parameter", RedundancyParameter::kQ7, "RedundancyParameter.Q7"),
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "options"_a = IKOptions(),
      "All analytical IK solutions for a target end-effector pose, resolving the redundancy by holding "
      "`parameter` at `redundancy_value`.\n\n"
      "Uses the nominal Franka geometry (no robot connection required). Up to 8 branches; solutions "
      "outside the joint limits are dropped.\n\n"
      "Args:\n"
      "    o_t_ee: Target pose of the end-effector frame relative to base.\n"
      "    redundancy_value: Value at which `parameter` is held [rad].\n"
      "    parameter: Which quantity resolves the redundancy.\n"
      "    f_t_ee: Transform from flange to end-effector frame. Default (identity) treats o_t_ee as the flange.\n"
      "    options: Joint limits and singularity/solver options.\n\n"
      "Returns:\n"
      "    A list of valid joint configurations [rad] (possibly empty).");

  m.def(
      "inverse_kinematics_nearest",
      [](const Affine &o_t_ee, const Vector7d &q_seed, const Affine &f_t_ee, const IKOptions &options,
         std::optional<double> redundancy_value, RedundancyParameter parameter,
         std::optional<double> max_distance) {
        std::optional<py::gil_scoped_release> release;
        if (parameter == RedundancyParameter::kSwivel) release.emplace();
        return inverseKinematicsNearest(o_t_ee, q_seed, f_t_ee, options, redundancy_value, parameter, max_distance);
      },
      "o_t_ee"_a,
      "q_seed"_a,
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "options"_a = IKOptions(),
      "redundancy_value"_a = std::nullopt,
      py::arg_v("parameter", RedundancyParameter::kQ7, "RedundancyParameter.Q7"),
      "max_distance"_a = std::nullopt,
      "The single IK solution nearest to `q_seed`, measured as the largest single-joint move.\n\n"
      "If `redundancy_value` is None, the seed's own value for `parameter` is held (e.g. the seed's joint 7), "
      "i.e. the current arm posture is kept.\n\n"
      "Nearest means nearest among the branches that reach the pose within the joint limits, and that set "
      "changes along a Cartesian path, so consecutive calls can switch branches and return a configuration "
      "far from the seed. Pass `max_distance` to get None instead of such a jump.\n\n"
      "Args:\n"
      "    o_t_ee: Target pose of the end-effector frame relative to base.\n"
      "    q_seed: Seed configuration to stay close to [rad].\n"
      "    f_t_ee: Transform from flange to end-effector frame.\n"
      "    options: Joint limits and singularity/solver options.\n"
      "    redundancy_value: Value at which `parameter` is held [rad], or None to hold the seed's value.\n"
      "    parameter: Which quantity resolves the redundancy.\n"
      "    max_distance: Largest acceptable single-joint move from `q_seed` [rad]. If the nearest branch is "
      "farther than this, None is returned instead. None (the default) accepts any distance.\n\n"
      "Returns:\n"
      "    The nearest valid configuration [rad], or None if none exists (or none is within `max_distance`).");
}
