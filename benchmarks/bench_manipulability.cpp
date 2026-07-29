// Benchmark for the manipulability nullspace task of the Cartesian impedance controller.
//
// Question this answers: the controller currently derives the manipulability gradient from a
// JacobiSVD pseudo-inverse plus seven franka::Model::pose() calls for the joint origins. Now that
// an analytical Jacobian and forward kinematics live in the tree (third_party/geofik), is a
// different formulation worth it?
//
// The benchmark is deliberately self-contained: it links Eigen and the vendored GeoFIK only, so it
// runs without libfranka and without a robot. That has one consequence worth stating up front:
// franka::Model::pose() cannot be timed here, because franka::Model needs the model library that
// libfranka downloads from the robot. Every variant below therefore sources its kinematics from
// GeoFIK, which makes the comparison between variants exact and makes the reported cost of the
// "current" variant a *lower bound* on what the shipped code costs (a cross-shared-object call into
// libfranka's generated model code cannot be cheaper than the inlined GeoFIK sweep it replaces).
//
// Conventions: franky/libfranka order the geometric Jacobian [linear; angular]; GeoFIK returns its
// transpose in [angular; linear] order. Everything below is converted to the franky convention so
// the "current" variant is a line-for-line port of src/motion/cartesian_impedance_base.cpp.

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "geofik.h"

// Defined in geofik.cpp but not declared in geofik.h: fills all nine frame transforms (T01..T0E)
// in a single forward sweep, which is what the joint-origin lookup actually needs.
void franka_fk_all_frames(std::array<Eigen::Matrix4d, 9> &Ts, const std::array<double, 7> &q);

namespace {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Jacobian = Eigen::Matrix<double, 6, 7>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// Panda joint limits, mirroring franky::kPandaJointLimits in src/kinematics.cpp.
const Vector7d kLower = (Vector7d() << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973).finished();
const Vector7d kUpper = (Vector7d() << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973).finished();

std::array<double, 7> toStd(const Vector7d &q) {
  std::array<double, 7> out{};
  for (int i = 0; i < 7; ++i) out[i] = q[i];
  return out;
}

// ---------------------------------------------------------------------------------------------
// Kinematics inputs, in the franky/libfranka [linear; angular] convention.
// ---------------------------------------------------------------------------------------------

struct Kinematics {
  Jacobian jacobian;                 // zero Jacobian at the EE frame, [v; omega]
  Eigen::Matrix<double, 3, 7> p;     // joint frame origins in base
  Eigen::Vector3d pn;                // EE origin in base
};

Jacobian jacobianFromGeofik(const Vector7d &q) {
  // GeoFIK hands back J^T with rows ordered [angular; linear]; flip both.
  const auto jt = J_from_q(toStd(q), 'E');
  Jacobian j;
  for (int c = 0; c < 7; ++c) {
    j(0, c) = jt[c][3];
    j(1, c) = jt[c][4];
    j(2, c) = jt[c][5];
    j(3, c) = jt[c][0];
    j(4, c) = jt[c][1];
    j(5, c) = jt[c][2];
  }
  return j;
}

Kinematics kinematics(const Vector7d &q) {
  Kinematics k;
  k.jacobian = jacobianFromGeofik(q);
  std::array<Eigen::Matrix4d, 9> ts{};
  franka_fk_all_frames(ts, toStd(q));
  for (int i = 0; i < 7; ++i) k.p.col(i) = ts[i].block<3, 1>(0, 3);
  k.pn = ts[8].block<3, 1>(0, 3);
  return k;
}

// The joint-origin lookup on its own; this is what the seven franka::Model::pose() calls buy.
void jointOriginsOneSweep(const Vector7d &q, Eigen::Matrix<double, 3, 7> &p, Eigen::Vector3d &pn) {
  std::array<Eigen::Matrix4d, 9> ts{};
  franka_fk_all_frames(ts, toStd(q));
  for (int i = 0; i < 7; ++i) p.col(i) = ts[i].block<3, 1>(0, 3);
  pn = ts[8].block<3, 1>(0, 3);
}

// Seven independent FK calls, mirroring the shipped loop over kJoint1..kJoint7. Each call redoes
// the chain from the base, so this is the O(n^2) shape of the current code.
void jointOriginsSevenCalls(const Vector7d &q, Eigen::Matrix<double, 3, 7> &p, Eigen::Vector3d &pn) {
  const auto qa = toStd(q);
  for (int i = 0; i < 7; ++i) p.col(i) = franka_fk(qa, static_cast<char>('1' + i)).block<3, 1>(0, 3);
  pn = franka_fk(qa, 'E').block<3, 1>(0, 3);
}

// ---------------------------------------------------------------------------------------------
// The shared piece: dJ/dq_i for a serial revolute chain. Every analytic variant needs it, so it is
// written once and the variants differ only in how they turn it into the gradient.
// ---------------------------------------------------------------------------------------------

// Fills dJ with the derivative of the Jacobian with respect to joint i, exactly as the shipped code
// builds it.
void jacobianDerivative(
    int i, const Eigen::Matrix<double, 3, 7> &z, const Eigen::Matrix<double, 3, 7> &p,
    const Eigen::Vector3d &pn, Jacobian &dJ) {
  dJ.setZero();
  const Eigen::Vector3d zi = z.col(i);
  for (int k = 0; k < 7; ++k) {
    const Eigen::Vector3d zk = z.col(k);
    if (k < i) {
      dJ.block<3, 1>(0, k) = zk.cross(zi.cross(pn - p.col(i)));
    } else {
      const Eigen::Vector3d r = pn - p.col(k);
      const Eigen::Vector3d zi_x_zk = zi.cross(zk);
      dJ.block<3, 1>(0, k) = zi_x_zk.cross(r) + zk.cross(zi.cross(r));
      dJ.block<3, 1>(3, k) = zi_x_zk;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Variant A: the shipped implementation, ported verbatim.
// ---------------------------------------------------------------------------------------------

double manipulabilityDet(const Jacobian &jacobian) {
  const double determinant = (jacobian * jacobian.transpose()).determinant();
  return std::sqrt(std::max(determinant, 0.0));
}

Vector7d gradientCurrent(const Kinematics &kin) {
  const Jacobian &jacobian = kin.jacobian;
  const double w = manipulabilityDet(jacobian);
  if (w < 1e-10) return Vector7d::Zero();

  Eigen::JacobiSVD<Jacobian> svd(jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 6> j_pinv = Eigen::Matrix<double, 7, 6>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (svd.singularValues()[i] > tolerance)
      j_pinv += (1.0 / svd.singularValues()[i]) * svd.matrixV().col(i) * svd.matrixU().col(i).transpose();
  }

  const auto z = jacobian.bottomRows<3>();

  Vector7d gradient = Vector7d::Zero();
  Jacobian dJ;
  for (int i = 0; i < 7; ++i) {
    jacobianDerivative(i, z, kin.p, kin.pn, dJ);
    gradient[i] = w * (j_pinv.transpose().array() * dJ.array()).sum();
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant B: same math, but the pseudo-inverse and the determinant both come out of one Cholesky
// factorization of J J^T instead of a full SVD.
//
// For full-row-rank J, J^# = J^T (J J^T)^{-1}, so (J^#)^T = (J J^T)^{-1} J -- which is exactly the
// matrix the trace contracts against. One LLT plus one 6x7 solve replaces the SVD, and
// w = sqrt(det(J J^T)) = prod(diag(L)) falls out of the same factorization for free.
// ---------------------------------------------------------------------------------------------

Vector7d gradientLLT(const Kinematics &kin) {
  const Jacobian &jacobian = kin.jacobian;
  const Eigen::LLT<Matrix6d> llt(jacobian * jacobian.transpose());
  if (llt.info() != Eigen::Success) return Vector7d::Zero();
  const double w = llt.matrixL().nestedExpression().diagonal().prod();
  if (w < 1e-10) return Vector7d::Zero();

  // a == (J^#)^T, 6x7.
  const Eigen::Matrix<double, 6, 7> a = llt.solve(jacobian);
  const auto z = jacobian.bottomRows<3>();

  Vector7d gradient = Vector7d::Zero();
  Jacobian dJ;
  for (int i = 0; i < 7; ++i) {
    jacobianDerivative(i, z, kin.p, kin.pn, dJ);
    gradient[i] = w * (a.array() * dJ.array()).sum();
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant C: variant B with the dJ matrix never materialized -- the trace is accumulated column by
// column straight out of the cross products.
// ---------------------------------------------------------------------------------------------

Vector7d gradientFused(const Kinematics &kin) {
  const Jacobian &jacobian = kin.jacobian;
  const Eigen::LLT<Matrix6d> llt(jacobian * jacobian.transpose());
  if (llt.info() != Eigen::Success) return Vector7d::Zero();
  const double w = llt.matrixL().nestedExpression().diagonal().prod();
  if (w < 1e-10) return Vector7d::Zero();

  const Eigen::Matrix<double, 6, 7> a = llt.solve(jacobian);
  const auto z = jacobian.bottomRows<3>();

  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    const Eigen::Vector3d ri = kin.pn - kin.p.col(i);
    const Eigen::Vector3d zi_x_ri = zi.cross(ri);
    double trace = 0.0;
    for (int k = 0; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      if (k < i) {
        trace += a.col(k).head<3>().dot(zk.cross(zi_x_ri));
      } else {
        const Eigen::Vector3d r = kin.pn - kin.p.col(k);
        const Eigen::Vector3d zi_x_zk = zi.cross(zk);
        trace += a.col(k).head<3>().dot(zi_x_zk.cross(r) + zk.cross(zi.cross(r)));
        trace += a.col(k).tail<3>().dot(zi_x_zk);
      }
    }
    gradient[i] = w * trace;
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant E: same trace, but factor J^T directly with a Householder QR so J J^T is never formed.
//
// J^T = Q R with R upper triangular 6x6, so J J^T = R^T R without ever multiplying J by itself.
// Then w = sqrt(det(R^T R)) = |prod(diag(R))|, and (J^#)^T = (J J^T)^{-1} J = R^{-1} Q^T, a single
// triangular solve. This is the variant that keeps the LLT's speed without inheriting the squared
// condition number that both det(J J^T) and a Cholesky of J J^T pay for.
// ---------------------------------------------------------------------------------------------

// Returns w and, in `a`, (J^#)^T == pinv(J^T).
double manipulabilityQR(const Jacobian &jacobian, Eigen::Matrix<double, 6, 7> &a) {
  Eigen::HouseholderQR<Eigen::Matrix<double, 7, 6>> qr(jacobian.transpose());
  const Eigen::Matrix<double, 6, 6> r = qr.matrixQR().topLeftCorner<6, 6>().template triangularView<Eigen::Upper>();
  const double w = std::abs(r.diagonal().prod());
  const Eigen::Matrix<double, 7, 6> q_thin = qr.householderQ() * Eigen::Matrix<double, 7, 6>::Identity();
  a = r.template triangularView<Eigen::Upper>().solve(q_thin.transpose());
  return w;
}

Vector7d gradientQR(const Kinematics &kin) {
  const Jacobian &jacobian = kin.jacobian;
  Eigen::Matrix<double, 6, 7> a;
  const double w = manipulabilityQR(jacobian, a);
  if (w < 1e-10) return Vector7d::Zero();

  const auto z = jacobian.bottomRows<3>();
  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    const Eigen::Vector3d zi_x_ri = zi.cross(kin.pn - kin.p.col(i));
    double trace = 0.0;
    for (int k = 0; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      if (k < i) {
        trace += a.col(k).head<3>().dot(zk.cross(zi_x_ri));
      } else {
        const Eigen::Vector3d r = kin.pn - kin.p.col(k);
        const Eigen::Vector3d zi_x_zk = zi.cross(zk);
        trace += a.col(k).head<3>().dot(zi_x_zk.cross(r) + zk.cross(zi.cross(r)));
        trace += a.col(k).tail<3>().dot(zi_x_zk);
      }
    }
    gradient[i] = w * trace;
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant F: rank-revealing complete orthogonal decomposition.
//
// Plain HouseholderQR is not rank-revealing: at an exact singularity R simply has a zero on the
// diagonal and the triangular solve blows up. Eigen's CompleteOrthogonalDecomposition is the
// idiomatic pseudo-inverse for this reason -- it pivots, exposes rank(), and degrades to the
// minimum-norm solution instead of infinities. This is what a library would ship.
// ---------------------------------------------------------------------------------------------

double manipulabilityCOD(const Jacobian &jacobian, Eigen::Matrix<double, 6, 7> &a, int &rank) {
  Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double, 7, 6>> cod(jacobian.transpose());
  rank = static_cast<int>(cod.rank());
  // Column pivoting only permutes columns, so |prod(diag(R))| is still |det| up to sign.
  const double w = std::abs(cod.matrixT().topLeftCorner<6, 6>().diagonal().prod());
  a = cod.pseudoInverse();
  return w;
}

Vector7d gradientCOD(const Kinematics &kin) {
  const Jacobian &jacobian = kin.jacobian;
  Eigen::Matrix<double, 6, 7> a;
  int rank = 0;
  const double w = manipulabilityCOD(jacobian, a, rank);
  if (w < 1e-10) return Vector7d::Zero();

  const auto z = jacobian.bottomRows<3>();
  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    const Eigen::Vector3d zi_x_ri = zi.cross(kin.pn - kin.p.col(i));
    double trace = 0.0;
    for (int k = 0; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      if (k < i) {
        trace += a.col(k).head<3>().dot(zk.cross(zi_x_ri));
      } else {
        const Eigen::Vector3d r = kin.pn - kin.p.col(k);
        const Eigen::Vector3d zi_x_zk = zi.cross(zk);
        trace += a.col(k).head<3>().dot(zi_x_zk.cross(r) + zk.cross(zi.cross(r)));
        trace += a.col(k).tail<3>().dot(zi_x_zk);
      }
    }
    gradient[i] = w * trace;
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant G: damped least squares (Tikhonov) with a Cholesky solve.
//
// The mainstream real-time idiom, and what franka_example_controllers' pseudoInverse() does by
// default: J^#_lambda = J^T (J J^T + lambda^2 I)^-1. The regularizer makes J J^T positive definite
// by construction, so the Cholesky is always safe and always fast -- damping buys back exactly the
// conditioning that forming J J^T destroyed. The cost is bias: the damped inverse is not the
// pseudo-inverse, so the gradient it produces is not the gradient of w.
// ---------------------------------------------------------------------------------------------

double manipulabilityDLS(const Jacobian &jacobian, double lambda, Eigen::Matrix<double, 6, 7> &a) {
  const Matrix6d jjt = jacobian * jacobian.transpose();
  const Eigen::LLT<Matrix6d> llt(jjt + lambda * lambda * Matrix6d::Identity());
  a = llt.solve(jacobian);
  // w itself is still taken from the undamped product; damping it would change what is being
  // maximized, not just how it is computed.
  const Eigen::LLT<Matrix6d> llt_undamped(jjt);
  if (llt_undamped.info() != Eigen::Success) return 0.0;
  return llt_undamped.matrixL().nestedExpression().diagonal().prod();
}

Vector7d gradientDLS(const Kinematics &kin, double lambda) {
  const Jacobian &jacobian = kin.jacobian;
  Eigen::Matrix<double, 6, 7> a;
  const double w = manipulabilityDLS(jacobian, lambda, a);
  if (w < 1e-10) return Vector7d::Zero();

  const auto z = jacobian.bottomRows<3>();
  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    const Eigen::Vector3d zi_x_ri = zi.cross(kin.pn - kin.p.col(i));
    double trace = 0.0;
    for (int k = 0; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      if (k < i) {
        trace += a.col(k).head<3>().dot(zk.cross(zi_x_ri));
      } else {
        const Eigen::Vector3d r = kin.pn - kin.p.col(k);
        const Eigen::Vector3d zi_x_zk = zi.cross(zk);
        trace += a.col(k).head<3>().dot(zi_x_zk.cross(r) + zk.cross(zi.cross(r)));
        trace += a.col(k).tail<3>().dot(zi_x_zk);
      }
    }
    gradient[i] = w * trace;
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Variant D: no analytic dJ at all -- central differences of w(q), with w evaluated from the
// analytic Jacobian. This is the variant the analytical Jacobian in the tree actually enables:
// 14 J_from_q calls and 14 Cholesky factorizations, no joint-origin lookup, no dJ formula.
// ---------------------------------------------------------------------------------------------

double manipulabilityLLT(const Jacobian &jacobian) {
  const Eigen::LLT<Matrix6d> llt(jacobian * jacobian.transpose());
  if (llt.info() != Eigen::Success) return 0.0;
  return llt.matrixL().nestedExpression().diagonal().prod();
}

Vector7d gradientFiniteDifference(const Vector7d &q, double h) {
  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    Vector7d qp = q, qm = q;
    qp[i] += h;
    qm[i] -= h;
    gradient[i] = (manipulabilityLLT(jacobianFromGeofik(qp)) - manipulabilityLLT(jacobianFromGeofik(qm))) / (2.0 * h);
  }
  return gradient;
}

// ---------------------------------------------------------------------------------------------
// Reference gradient: Richardson-extrapolated central differences, O(h^4). Used only to score
// accuracy, never timed.
// ---------------------------------------------------------------------------------------------

Vector7d gradientReference(const Vector7d &q) {
  constexpr double h = 1e-4;
  Vector7d coarse = gradientFiniteDifference(q, h);
  Vector7d fine = gradientFiniteDifference(q, h / 2.0);
  return (4.0 * fine - coarse) / 3.0;
}

// ---------------------------------------------------------------------------------------------
// The surrounding nullspace block, which shares the same factorization.
//
// computeCommand() builds the nullspace projector with a *second* SVD --
// pseudoInverse(jacobian.transpose()) -- whenever any nullspace task is active. But
// pinv(J^T) == (J J^T)^{-1} J, the very matrix variant B already solved for. These two functions
// measure the whole projector-plus-manipulability block both ways.
// ---------------------------------------------------------------------------------------------

Eigen::Matrix<double, 6, 7> pseudoInverseSVD(const Eigen::Matrix<double, 7, 6> &matrix) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &singular_values = svd.singularValues();
  Eigen::Matrix<double, 6, 7> sigma_pinv = Eigen::Matrix<double, 6, 7>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) sigma_pinv(i, i) = 1.0 / singular_values[i];
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}

Vector7d nullspaceBlockCurrent(const Kinematics &kin, const Vector7d &tau_other) {
  const Jacobian &j = kin.jacobian;
  const auto jt_pinv = pseudoInverseSVD(j.transpose());
  const Eigen::Matrix<double, 7, 7> projector =
      Eigen::Matrix<double, 7, 7>::Identity() - j.transpose() * jt_pinv;
  return projector.transpose() * (gradientCurrent(kin) + tau_other);
}

Vector7d nullspaceBlockShared(const Kinematics &kin, const Vector7d &tau_other) {
  const Jacobian &j = kin.jacobian;
  const Eigen::LLT<Matrix6d> llt(j * j.transpose());
  if (llt.info() != Eigen::Success) return Vector7d::Zero();
  const double w = llt.matrixL().nestedExpression().diagonal().prod();
  const Eigen::Matrix<double, 6, 7> a = llt.solve(j);  // == pinv(J^T) == (J^#)^T
  const Eigen::Matrix<double, 7, 7> projector = Eigen::Matrix<double, 7, 7>::Identity() - j.transpose() * a;

  Vector7d gradient = Vector7d::Zero();
  if (w >= 1e-10) {
    const auto z = j.bottomRows<3>();
    for (int i = 0; i < 7; ++i) {
      const Eigen::Vector3d zi = z.col(i);
      const Eigen::Vector3d zi_x_ri = zi.cross(kin.pn - kin.p.col(i));
      double trace = 0.0;
      for (int k = 0; k < 7; ++k) {
        const Eigen::Vector3d zk = z.col(k);
        if (k < i) {
          trace += a.col(k).head<3>().dot(zk.cross(zi_x_ri));
        } else {
          const Eigen::Vector3d r = kin.pn - kin.p.col(k);
          const Eigen::Vector3d zi_x_zk = zi.cross(zk);
          trace += a.col(k).head<3>().dot(zi_x_zk.cross(r) + zk.cross(zi.cross(r)));
          trace += a.col(k).tail<3>().dot(zi_x_zk);
        }
      }
      gradient[i] = w * trace;
    }
  }
  return projector.transpose() * (gradient + tau_other);
}

// ---------------------------------------------------------------------------------------------
// Timing harness.
// ---------------------------------------------------------------------------------------------

struct Stats {
  double median_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  double mean_us = 0.0;
};

Stats summarize(std::vector<double> &samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  Stats s;
  const size_t n = samples_us.size();
  s.median_us = samples_us[n / 2];
  s.p99_us = samples_us[static_cast<size_t>(0.99 * static_cast<double>(n - 1))];
  s.max_us = samples_us.back();
  double sum = 0.0;
  for (double v : samples_us) sum += v;
  s.mean_us = sum / static_cast<double>(n);
  return s;
}

// Times fn once per configuration, repeated `reps` times, one clock read per call.
template <typename Fn>
Stats timeCalls(const std::vector<Vector7d> &qs, const std::vector<Kinematics> &kins, int reps, Fn &&fn) {
  volatile double sink = 0.0;
  for (size_t i = 0; i < qs.size(); ++i) sink += fn(qs[i], kins[i]);  // warm up

  std::vector<double> samples;
  samples.reserve(qs.size() * static_cast<size_t>(reps));
  for (int r = 0; r < reps; ++r) {
    for (size_t i = 0; i < qs.size(); ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const double v = fn(qs[i], kins[i]);
      const auto t1 = std::chrono::steady_clock::now();
      sink += v;
      samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  (void)sink;
  return summarize(samples);
}

void printRow(const char *name, const Stats &s, const Stats *baseline) {
  char speedup[32] = "";
  if (baseline != nullptr && s.median_us > 0.0) {
    std::snprintf(speedup, sizeof(speedup), "%6.1fx", baseline->median_us / s.median_us);
  }
  std::printf("  %-42s %9.3f %9.3f %9.3f  %s\n", name, s.median_us, s.p99_us, s.max_us, speedup);
}

void printHeader(const char *title) {
  std::printf("\n%s\n", title);
  std::printf("  %-42s %9s %9s %9s  %s\n", "variant", "med (us)", "p99 (us)", "max (us)", "vs base");
  std::printf("  %s\n", std::string(84, '-').c_str());
}

// ---------------------------------------------------------------------------------------------

// Clock resolution and call overhead, so single-call numbers can be read honestly.
void reportClockOverhead() {
  std::vector<double> samples;
  samples.reserve(20000);
  for (int i = 0; i < 20000; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  const Stats s = summarize(samples);
  std::printf("clock: back-to-back steady_clock read costs %.4f us (median), %.4f us (p99)\n", s.median_us, s.p99_us);
}

}  // namespace

int main(int argc, char **argv) {
  int n_configs = 2048;
  int reps = 20;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--configs") == 0 && i + 1 < argc) n_configs = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
  }

  std::mt19937 rng(20240729);
  std::vector<Vector7d> qs;
  std::vector<Kinematics> kins;
  qs.reserve(static_cast<size_t>(n_configs));
  kins.reserve(static_cast<size_t>(n_configs));
  // Sample inside the limits with a small inset, so no sample sits exactly on a limit.
  for (int c = 0; c < n_configs; ++c) {
    Vector7d q;
    for (int i = 0; i < 7; ++i) {
      std::uniform_real_distribution<double> dist(kLower[i] + 0.05, kUpper[i] - 0.05);
      q[i] = dist(rng);
    }
    qs.push_back(q);
    kins.push_back(kinematics(q));
  }

  std::printf("franky manipulability benchmark\n");
  std::printf("configurations: %d, repetitions: %d, samples per variant: %d\n", n_configs, reps, n_configs * reps);
  reportClockOverhead();

  // -------------------------------------------------------------------------------------------
  // 1. Gradient variants, kinematics precomputed (isolates the gradient algebra alone).
  // -------------------------------------------------------------------------------------------
  printHeader("1. manipulability gradient, Jacobian and joint origins supplied");
  const Stats base = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientCurrent(k).sum();
  });
  printRow("A. current (JacobiSVD pinv + dJ)", base, nullptr);

  const Stats llt = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientLLT(k).sum();
  });
  printRow("B. LLT pinv + dJ", llt, &base);

  const Stats fused = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientFused(k).sum();
  });
  printRow("C. LLT pinv + fused dJ (no dJ matrix)", fused, &base);

  const Stats fd = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &) {
    return gradientFiniteDifference(q, 1e-5).sum();
  });
  printRow("D. central differences via analytic J (14x)", fd, &base);

  const Stats qr = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientQR(k).sum();
  });
  printRow("E. QR of J^T + fused dJ (no J J^T formed)", qr, &base);

  const Stats cod = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientCOD(k).sum();
  });
  printRow("F. COD (rank-revealing) + fused dJ", cod, &base);

  const Stats dls = timeCalls(qs, kins, reps, [](const Vector7d &, const Kinematics &k) {
    return gradientDLS(k, 1e-2).sum();
  });
  printRow("G. damped LS + LLT + fused dJ (lambda=1e-2)", dls, &base);

  // -------------------------------------------------------------------------------------------
  // 2. Kinematics sourcing: what the seven pose() calls cost, and the one-sweep alternative.
  // -------------------------------------------------------------------------------------------
  printHeader("2. kinematics sourcing (GeoFIK; libfranka's Model::pose is not measurable offline)");
  const Stats seven = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &) {
    Eigen::Matrix<double, 3, 7> p;
    Eigen::Vector3d pn;
    jointOriginsSevenCalls(q, p, pn);
    return p.sum() + pn.sum();
  });
  printRow("joint origins, 7 independent FK calls", seven, nullptr);

  const Stats sweep = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &) {
    Eigen::Matrix<double, 3, 7> p;
    Eigen::Vector3d pn;
    jointOriginsOneSweep(q, p, pn);
    return p.sum() + pn.sum();
  });
  printRow("joint origins, 1 forward sweep", sweep, &seven);

  const Stats jac = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &) {
    return jacobianFromGeofik(q).sum();
  });
  printRow("analytic Jacobian (J_from_q)", jac, &seven);

  // -------------------------------------------------------------------------------------------
  // 3. End to end: gradient including its own kinematics.
  // -------------------------------------------------------------------------------------------
  printHeader("3. gradient including kinematics sourcing");
  const Stats e2e_current = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &k) {
    Kinematics local;
    local.jacobian = k.jacobian;  // in the shipped code the Jacobian is already in hand
    jointOriginsSevenCalls(q, local.p, local.pn);
    return gradientCurrent(local).sum();
  });
  printRow("A. current shape (7 FK + SVD + dJ)", e2e_current, nullptr);

  const Stats e2e_fused = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &k) {
    Kinematics local;
    local.jacobian = k.jacobian;
    jointOriginsOneSweep(q, local.p, local.pn);
    return gradientFused(local).sum();
  });
  printRow("C. 1 sweep + LLT + fused dJ", e2e_fused, &e2e_current);

  const Stats e2e_fd = timeCalls(qs, kins, reps, [](const Vector7d &q, const Kinematics &) {
    return gradientFiniteDifference(q, 1e-5).sum();
  });
  printRow("D. central differences (no kinematics needed)", e2e_fd, &e2e_current);

  // -------------------------------------------------------------------------------------------
  // 4. The whole nullspace block, where the projector's second SVD lives.
  // -------------------------------------------------------------------------------------------
  printHeader("4. full nullspace block (projector + manipulability torque)");
  const Vector7d tau_other = Vector7d::Constant(0.1);
  const Stats block_current = timeCalls(qs, kins, reps, [&](const Vector7d &, const Kinematics &k) {
    return nullspaceBlockCurrent(k, tau_other).sum();
  });
  printRow("A. current (2 JacobiSVDs)", block_current, nullptr);

  const Stats block_shared = timeCalls(qs, kins, reps, [&](const Vector7d &, const Kinematics &k) {
    return nullspaceBlockShared(k, tau_other).sum();
  });
  printRow("B. one LLT shared by projector + gradient", block_shared, &block_current);

  const Stats block_qr = timeCalls(qs, kins, reps, [&](const Vector7d &, const Kinematics &k) {
    // The QR gives pinv(J^T) and the manipulability pinv from the same factorization too.
    Eigen::Matrix<double, 6, 7> a;
    const double w = manipulabilityQR(k.jacobian, a);
    (void)w;
    const Eigen::Matrix<double, 7, 7> projector =
        Eigen::Matrix<double, 7, 7>::Identity() - k.jacobian.transpose() * a;
    return (projector.transpose() * (gradientQR(k) + tau_other)).sum();
  });
  printRow("C. one QR shared by projector + gradient", block_qr, &block_current);

  // -------------------------------------------------------------------------------------------
  // 5. Accuracy against a Richardson-extrapolated reference.
  // -------------------------------------------------------------------------------------------
  std::printf("\n5. accuracy vs Richardson-extrapolated central differences (O(h^4))\n");
  std::printf("  %-42s %12s %12s\n", "variant", "med rel err", "max rel err");
  std::printf("  %s\n", std::string(68, '-').c_str());

  const int n_accuracy = std::min(n_configs, 512);
  std::vector<double> err_current, err_fused, err_fd, err_qr, err_cod, err_dls, err_dls_small;
  for (int c = 0; c < n_accuracy; ++c) {
    const Vector7d ref = gradientReference(qs[c]);
    const double scale = std::max(ref.norm(), 1e-12);
    err_current.push_back((gradientCurrent(kins[c]) - ref).norm() / scale);
    err_fused.push_back((gradientFused(kins[c]) - ref).norm() / scale);
    err_fd.push_back((gradientFiniteDifference(qs[c], 1e-5) - ref).norm() / scale);
    err_qr.push_back((gradientQR(kins[c]) - ref).norm() / scale);
    err_cod.push_back((gradientCOD(kins[c]) - ref).norm() / scale);
    err_dls_small.push_back((gradientDLS(kins[c], 1e-3) - ref).norm() / scale);
    err_dls.push_back((gradientDLS(kins[c], 1e-2) - ref).norm() / scale);
  }
  auto report = [](const char *name, std::vector<double> &e) {
    std::sort(e.begin(), e.end());
    std::printf("  %-42s %12.3e %12.3e\n", name, e[e.size() / 2], e.back());
  };
  report("A. current (JacobiSVD pinv + dJ)", err_current);
  report("C. LLT pinv + fused dJ", err_fused);
  report("D. central differences, h=1e-5", err_fd);
  report("E. QR pinv + fused dJ", err_qr);
  report("F. COD pinv + fused dJ", err_cod);
  report("G. damped LS, lambda=1e-3", err_dls_small);
  report("G. damped LS, lambda=1e-2", err_dls);

  // -------------------------------------------------------------------------------------------
  // 6. Conditioning: how each variant behaves as the arm approaches a singularity.
  //
  // Rather than guess at a singular pose, descend on w with the analytic gradient until the arm is
  // genuinely rank-deficient, and score both routes to w against the product of the singular
  // values, which is what w is by definition and the numerically well-behaved way to get it.
  //
  // The point of interest: the shipped code forms det(J J^T) explicitly and then square-roots it,
  // so the relative error in w is half the relative error in a quantity of size w^2 -- the accuracy
  // collapses once w^2 approaches the rounding level of the 6x6 product. The Cholesky diagonal
  // gives w as a product of factors of size ~w^(1/6), and never squares anything.
  // -------------------------------------------------------------------------------------------
  std::printf("\n6. near-singular behaviour (descent on w toward a rank-deficient configuration)\n");
  std::printf("  %-8s %13s %11s %11s %11s %8s %10s %10s\n", "w decade", "w (sigmas)", "det rel", "chol rel", "qr rel",
              "llt ok", "cod rank", "dls g err");
  std::printf("  %s\n", std::string(94, '-').c_str());
  int llt_failures = 0;
  {
    Vector7d q;
    q << 0.0, -0.3, 0.0, -1.8, 0.0, 1.2, 0.7854;
    double next_decade = 1e30;
    for (int step = 0; step <= 20000; ++step) {
      const Kinematics k = kinematics(q);
      // Reference: w == prod(sigma_i(J)), taken from the singular values of J itself. This never
      // forms J J^T, so it does not square the condition number.
      Eigen::JacobiSVD<Jacobian> svd(k.jacobian);
      const double w_ref = svd.singularValues().prod();

      const Eigen::LLT<Matrix6d> llt(k.jacobian * k.jacobian.transpose());
      const bool llt_ok = llt.info() == Eigen::Success;
      if (!llt_ok) ++llt_failures;

      if (w_ref < next_decade) {  // roughly one line per decade of w
        const double w_det = manipulabilityDet(k.jacobian);
        const double w_chol = manipulabilityLLT(k.jacobian);
        Eigen::Matrix<double, 6, 7> a;
        const double w_qr = manipulabilityQR(k.jacobian, a);
        int rank = 0;
        Eigen::Matrix<double, 6, 7> a_cod;
        manipulabilityCOD(k.jacobian, a_cod, rank);
        // How far the damped gradient has drifted from the true one at this conditioning.
        const Vector7d g_true = gradientQR(k);
        const Vector7d g_dls = gradientDLS(k, 1e-2);
        const double dls_err = (g_dls - g_true).norm() / std::max(g_true.norm(), 1e-300);
        const double scale = std::max(w_ref, 1e-300);
        std::printf(
            "  %-8.0e %13.6e %11.2e %11.2e %11.2e %8s %10d %10.2e\n", w_ref, w_ref, std::abs(w_det - w_ref) / scale,
            std::abs(w_chol - w_ref) / scale, std::abs(w_qr - w_ref) / scale, llt_ok ? "yes" : "NO", rank, dls_err);
        next_decade = w_ref / 10.0;
      }
      if (w_ref < 1e-12) break;

      // Descend on w. The gradient vanishes with w, so normalize the step.
      const Vector7d g = gradientQR(k);
      const double gn = g.norm();
      if (gn < 1e-300) break;
      q -= 0.01 * g / gn;
      q = q.array().max(kLower.array() + 1e-6).min(kUpper.array() - 1e-6).matrix();
    }
  }
  std::printf("  Cholesky of J J^T failed on %d of the configurations along the descent\n", llt_failures);

  std::printf("\n");
  return 0;
}
