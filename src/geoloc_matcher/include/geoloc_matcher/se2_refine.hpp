// SE(2) refinement (T17): bring the coarse phase-correlation fix to an accurate
// pose.
//
// The phase-correlation channel (phase_corr.hpp) localises a patch inside a
// prior map window in one global shot, but its accuracy is deliberately coarse
// (5-15 m / 1-3 deg -- see task card T19): the rotation comes from a log-polar
// Fourier-Mellin step and the translation from a parabolic peak fit, both of
// which leave a residual of a fraction of a pixel and a fraction of a degree.
//
// This module closes that gap. It takes the coarse SE(2) hypothesis as an
// initialisation and drives it to sub-pixel accuracy with a dense, gradient
// based inverse search: a forward-compositional Lucas-Kanade Gauss-Newton
// minimisation over the three SE(2) parameters (dx, dy, dtheta), with a
// backtracking line search for robustness. No descriptor extraction, no scale
// DoF -- scale is known from AGL and the DSM (ADR-002), so adding it would only
// drag the other parameters around a degenerate direction on a near-planar
// scene.
//
// The working field is the intensity with its global mean removed. Mean
// subtraction gives DC-offset invariance (the dominant illumination difference
// between a camera frame and a satellite orthophoto) while staying a *global*
// operation: it leaves the residual a constant at the true pose of a
// self-consistent "patch is a crop of the map" pair, which a gradient-based
// search is insensitive to -- so the minimum is not shifted the way a per-image
// variance normalisation or a box high-pass would shift it. The remaining
// outliers (occlusions, moving objects, shadows, provider seams, residual
// contrast) are absorbed by a Tukey biweight on the residual.
//
// This channel STILL never judges. It produces a refined pose plus quality
// metrics and leaves the accept/reject decision to geoloc_integrity, exactly as
// the coarse channel does.
//
// Conventions (identical to phase_corr.hpp so the two stages compose without
// sign bugs):
//   * `delta_yaw` is the rotation TO APPLY to the patch, radians, image frame.
//   * `shift_east_px` is +x (columns, east), `shift_north_px` is +y (north, i.e.
//     -rows, since row increases southward).
//   * The patch is placed in the map with its CENTRE at
//       (shift_east_px + (w-1)/2, -shift_north_px + (h-1)/2)
//     and rotated by delta_yaw. The forward warp from a patch pixel to its map
//     coordinate is therefore
//       map = centre + R(theta) * (patch - patch_centre),  R = [[c,-s],[s,c]].

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "geoloc_common/angles.hpp"
#include "geoloc_common/covariance.hpp"
#include "geoloc_matcher/phase_corr.hpp"

namespace geoloc {

struct Se2RefineConfig {
  int max_iterations = 30;
  double trans_tol = 1e-2;  // stop when |dtx|,|dty| below this (px)
  double rot_tol = 1e-4;    // stop when |dtheta| below this (rad)
  double tukey_c = 4.685;        // robust-loss constant (95% efficiency, Gaussian)
  double robust_sigma = 0.02;    // floor on the adaptive robust scale (intensity)
  double inlier_thresh = 0.1;    // |residual| below this (intensity) => inlier
  double lambda = 1e-6;          // Levenberg-Marquardt damping (relative to max |diag|)
  double min_overlap_frac = 0.5; // fraction of the patch that must map inside the map
  double confidence_thresh = 0.5;  // patch pixels with confidence <= this are excluded
};

struct Se2RefineResult {
  bool success = false;
  bool converged = false;
  int iterations = 0;
  double shift_east_px = 0.0;
  double shift_north_px = 0.0;
  double delta_yaw = 0.0;  // rad, rotation to apply to patch (phase-corr convention)
  double residual_rms = 0.0;  // photometric RMS over inliers (mean-subtracted intensity)
  double cost = 0.0;          // final robust cost (sum of Tukey rho)
  int n_inliers = 0;
  int n_total = 0;
  double inlier_ratio = 0.0;
  double mean_confidence = 0.0;
  // Provisional covariance in (east_px, north_px, yaw_rad) from the Gauss-Newton
  // information matrix. Converted to metres by se2RefineToFix.
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
};

namespace se2refine_detail {

/// Central-difference gradient of a raw row-major field. Unlike a Sobel kernel,
/// a central difference has unit gain for a ramp, so its magnitude matches the
/// derivative of the pointwise residual -- which is what the Lucas-Kanade
/// Jacobian must use. (Sobel's response is 8x for a ramp and would shrink every
/// step by 8x.)
inline void centralGradient(const std::vector<double>& img, int w, int h,
                            std::vector<double>& gx, std::vector<double>& gy) {
  gx.assign(img.size(), 0.0);
  gy.assign(img.size(), 0.0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int xm = std::max(0, x - 1), xp = std::min(w - 1, x + 1);
      const int ym = std::max(0, y - 1), yp = std::min(h - 1, y + 1);
      gx[static_cast<size_t>(y) * w + x] =
          0.5 * (img[static_cast<size_t>(y) * w + xp] - img[static_cast<size_t>(y) * w + xm]);
      gy[static_cast<size_t>(y) * w + x] =
          0.5 * (img[static_cast<size_t>(yp) * w + x] - img[static_cast<size_t>(ym) * w + x]);
    }
  }
}

/// Bilinear sample of a scalar field; returns false when out of bounds.
inline bool sampleBilinear(const std::vector<double>& f, int w, int h, double u, double v,
                           double& out) {
  if (u < 0.0 || v < 0.0 || u > static_cast<double>(w - 1) ||
      v > static_cast<double>(h - 1)) {
    return false;
  }
  const int x0 = static_cast<int>(std::floor(u));
  const int y0 = static_cast<int>(std::floor(v));
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const double fx = u - x0, fy = v - y0;
  const double i00 = f[static_cast<size_t>(y0) * w + x0];
  const double i10 = f[static_cast<size_t>(y0) * w + x1];
  const double i01 = f[static_cast<size_t>(y1) * w + x0];
  const double i11 = f[static_cast<size_t>(y1) * w + x1];
  out = i00 * (1 - fx) * (1 - fy) + i10 * fx * (1 - fy) + i01 * (1 - fx) * fy +
        i11 * fx * fy;
  return true;
}

inline double tukeyWeight(double r, double c) {
  if (std::abs(r) >= c) return 0.0;
  const double x = 1.0 - (r / c) * (r / c);
  return x * x;
}

inline double tukeyRho(double r, double c) {
  if (std::abs(r) >= c) return 1.0;
  const double x = 1.0 - (r / c) * (r / c);
  return 1.0 - x * x * x;
}

/// Plain (signed) median -- the robust DC offset of a residual field.
inline double medianOf(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::vector<double> a = v;
  const size_t k = a.size() / 2;
  std::nth_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(k), a.end());
  return a[k];
}

}  // namespace se2refine_detail

/// Dense SE(2) refiner. Reusable; owns no state beyond its config (scratch
/// buffers are per-call, matching the phase-correlation channel's allocation
/// pattern -- the zero-allocation pass is T32).
class Se2Refiner {
 public:
  explicit Se2Refiner(const Se2RefineConfig& cfg = Se2RefineConfig{}) : cfg_(cfg) {}

  Se2RefineResult refine(const GrayImage& patch, const GrayImage& ref,
                         double coarse_shift_east_px, double coarse_shift_north_px,
                         double coarse_delta_yaw,
                         const GrayImage* confidence = nullptr) const {
    using namespace se2refine_detail;
    Se2RefineResult res;
    const int pw = patch.width, ph = patch.height;
    const int rw = ref.width, rh = ref.height;
    if (patch.empty() || ref.empty() || pw > rw || ph > rh) return res;

    const double cx = (pw - 1) / 2.0, cy = (ph - 1) / 2.0;
    const int min_valid = static_cast<int>(cfg_.min_overlap_frac * pw * ph);

    // Working fields: intensity with the global mean removed (DC-offset
    // invariance). The Jacobian uses the central-difference gradient of the map
    // field, whose gain matches the pointwise residual derivative.
    std::vector<double> Fp = patch.data;
    std::vector<double> Fm = ref.data;
    auto meanSubtract = [](std::vector<double>& f) {
      double m = 0.0;
      for (double v : f) m += v;
      m /= static_cast<double>(f.size());
      for (double& v : f) v -= m;
    };
    meanSubtract(Fp);
    meanSubtract(Fm);
    std::vector<double> Gx, Gy;
    centralGradient(Fm, rw, rh, Gx, Gy);

    double tx = coarse_shift_east_px + cx;
    double ty = -coarse_shift_north_px + cy;
    double theta = coarse_delta_yaw;

    std::vector<double> rs;
    rs.reserve(static_cast<size_t>(pw) * ph);

    // Evaluate the robust cost and the Gauss-Newton normal equations at a given
    // pose. Returns the number of valid (mapped) pixels.
    auto evaluate = [&](double tx_, double ty_, double th_, double& cost, Eigen::Matrix3d& H,
                        Eigen::Vector3d& b, std::vector<double>& residuals) {
      const double ct = std::cos(th_), st = std::sin(th_);
      residuals.clear();
      for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
          if (confidence != nullptr && confidence->at(x, y) <= cfg_.confidence_thresh) continue;
          const double du = x - cx, dv = y - cy;
          const double u = tx_ + ct * du - st * dv;
          const double v = ty_ + st * du + ct * dv;
          double tv = 0.0;
          if (!sampleBilinear(Fm, rw, rh, u, v, tv)) continue;
          residuals.push_back(Fp[static_cast<size_t>(y) * pw + x] - tv);
        }
      if (residuals.empty()) {
        cost = std::numeric_limits<double>::infinity();
        H.setZero();
        b.setZero();
        return 0;
      }
      // Robust DC offset: the patch is a subregion of the map, so the residual
      // carries a constant offset (their means differ) that would otherwise
      // corrupt the adaptive scale. Remove the residual median, then adapt the
      // scale to the centred residual (MAD), floored so it never collapses.
      const double bias = medianOf(residuals);
      std::vector<double> cen = residuals;
      for (double& r : cen) r = std::abs(r - bias);
      const size_t k = cen.size() / 2;
      std::nth_element(cen.begin(), cen.begin() + static_cast<std::ptrdiff_t>(k), cen.end());
      const double sigma = std::max(1.4826 * cen[k], cfg_.robust_sigma);
      H.setZero();
      b.setZero();
      cost = 0.0;
      size_t idx = 0;
      for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
          if (confidence != nullptr && confidence->at(x, y) <= cfg_.confidence_thresh) continue;
          const double du = x - cx, dv = y - cy;
          const double u = tx_ + ct * du - st * dv;
          const double v = ty_ + st * du + ct * dv;
          double tv = 0.0;
          if (!sampleBilinear(Fm, rw, rh, u, v, tv)) continue;
          const double e = residuals[idx++] - bias;
          double gxs = 0.0, gys = 0.0;
          sampleBilinear(Gx, rw, rh, u, v, gxs);
          sampleBilinear(Gy, rw, rh, u, v, gys);
          const double dudth = -st * du - ct * dv;
          const double dvdth = ct * du - st * dv;
          const double J0 = -gxs;
          const double J1 = -gys;
          const double J2 = -gxs * dudth - gys * dvdth;
          const double w = tukeyWeight(e / sigma, cfg_.tukey_c);
          H(0, 0) += w * J0 * J0;
          H(0, 1) += w * J0 * J1;
          H(0, 2) += w * J0 * J2;
          H(1, 1) += w * J1 * J1;
          H(1, 2) += w * J1 * J2;
          H(2, 2) += w * J2 * J2;
          b(0) += w * J0 * e;
          b(1) += w * J1 * e;
          b(2) += w * J2 * e;
          cost += tukeyRho(e / sigma, cfg_.tukey_c);
        }
      H(1, 0) = H(0, 1);
      H(2, 0) = H(0, 2);
      H(2, 1) = H(1, 2);
      return static_cast<int>(residuals.size());
    };

    double cost = 0.0;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    bool converged = false;
    int iter = 0;

    for (; iter < cfg_.max_iterations; ++iter) {
      const int n = evaluate(tx, ty, theta, cost, H, b, rs);
      if (n < min_valid) return res;  // not enough overlap: honest refusal

      Eigen::Matrix3d Hlm = H;
      const double diag_max = std::max(1.0, H.diagonal().cwiseAbs().maxCoeff());
      Hlm.diagonal().array() += cfg_.lambda * diag_max;
      const Eigen::Vector3d dp = Hlm.ldlt().solve(-b);
      if (!dp.allFinite()) break;

      if (std::abs(dp(0)) < cfg_.trans_tol && std::abs(dp(1)) < cfg_.trans_tol &&
          std::abs(dp(2)) < cfg_.rot_tol) {
        converged = true;
        break;
      }

      // Backtracking line search: take the largest step that reduces the cost.
      double step = 1.0;
      bool moved = false;
      double nc = 0.0;
      Eigen::Matrix3d Hh;
      Eigen::Vector3d bh;
      while (step > 0.03) {
        const double ntx = tx + step * dp(0);
        const double nty = ty + step * dp(1);
        const double nth = normalizeAngle(theta + step * dp(2));
        const int nn = evaluate(ntx, nty, nth, nc, Hh, bh, rs);
        if (nn >= min_valid && nc < cost) {
          moved = true;
          break;
        }
        step *= 0.5;
      }
      if (!moved) {
        // No step reduces the cost: we are at a stationary point.
        converged = true;
        break;
      }
      tx += step * dp(0);
      ty += step * dp(1);
      theta = normalizeAngle(theta + step * dp(2));
    }

    // Final evaluation at the converged pose.
    const int n_final = evaluate(tx, ty, theta, cost, H, b, rs);
    if (n_final < min_valid) return res;

    const double bias = medianOf(rs);
    std::vector<double> cen = rs;
    for (double& r : cen) r = std::abs(r - bias);
    const double sigma_final = std::max(medianOf(cen), cfg_.robust_sigma);

    // Inliers / residual RMS / mean confidence (single pass over the patch).
    const double ct = std::cos(theta), st = std::sin(theta);
    int n_inliers = 0;
    double sum_sq = 0.0, conf_sum = 0.0;
    for (int y = 0; y < ph; ++y)
      for (int x = 0; x < pw; ++x) {
        if (confidence != nullptr && confidence->at(x, y) <= cfg_.confidence_thresh) continue;
        const double du = x - cx, dv = y - cy;
        const double u = tx + ct * du - st * dv;
        const double v = ty + st * du + ct * dv;
        double tv = 0.0;
        if (!sampleBilinear(Fm, rw, rh, u, v, tv)) continue;
        const double e = Fp[static_cast<size_t>(y) * pw + x] - tv - bias;
        if (std::abs(e) < cfg_.inlier_thresh) {
          ++n_inliers;
          sum_sq += e * e;
          conf_sum += confidence != nullptr ? confidence->at(x, y) : 1.0;
        }
      }

    res.success = true;
    res.converged = converged;
    res.iterations = iter;
    res.shift_east_px = tx - cx;
    res.shift_north_px = cy - ty;
    res.delta_yaw = normalizeAngle(theta);
    res.n_total = n_final;
    res.n_inliers = n_inliers;
    res.inlier_ratio = n_final > 0 ? static_cast<double>(n_inliers) / n_final : 0.0;
    res.residual_rms = n_inliers > 0 ? std::sqrt(sum_sq / n_inliers) : 0.0;
    res.mean_confidence = n_inliers > 0 ? conf_sum / n_inliers : 0.0;
    res.cost = cost;

    // Covariance = sigma^2 * H^-1 in (tx px, ty px, yaw rad). The residual scale
    // is re-estimated at the final pose; convert ty to the north axis
    // (north = -row) by flipping the ty row/column.
    {
      Eigen::Matrix3d Hlm = H;
      const double diag_max = std::max(1.0, H.diagonal().cwiseAbs().maxCoeff());
      Hlm.diagonal().array() += cfg_.lambda * diag_max;
      Eigen::Matrix3d Ctt =
          Hlm.ldlt().solve(Eigen::Matrix3d::Identity()) * (sigma_final * sigma_final);
      Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
      S(1, 1) = -1.0;  // north = -row
      Eigen::Matrix3d C = symmetrize(S * Ctt * S);
      if (isValidCovariance(C)) {
        res.covariance = C;
      } else {
        res.covariance = Eigen::Matrix3d::Zero();
        res.covariance(0, 0) = 1.0;
        res.covariance(1, 1) = 1.0;
        res.covariance(2, 2) = 1e-4;
      }
    }

    return res;
  }

 private:
  Se2RefineConfig cfg_;
};

/// Convert a refined result to SE2Fix-shaped fields (same struct as the coarse
/// channel so the node can drop it straight into the message). Position goes
/// from pixels to metres via the patch GSD; the pixel/yaw covariance from the
/// Hessian is scaled to metres and passed through the same systematic + floor
/// stages as the coarse channel.
inline PhaseCorrFix se2RefineToFix(const Se2RefineResult& r, double gsd, double peak_ratio,
                                   double scale_check, double covisibility,
                                   const PhaseCorrCovarianceConfig& cov =
                                       PhaseCorrCovarianceConfig{}) {
  PhaseCorrFix f;
  f.delta_east = r.shift_east_px * gsd;
  f.delta_north = r.shift_north_px * gsd;
  // Same sign convention as phaseCorrToFix: +theta in image space is -theta in
  // ENU yaw (CCW from East), because rows increase southward.
  f.delta_yaw = normalizeAngle(-r.delta_yaw);

  f.n_correspondences = static_cast<uint32_t>(r.n_total);
  f.n_inliers = static_cast<uint32_t>(r.n_inliers);
  f.inlier_ratio = static_cast<float>(r.inlier_ratio);
  f.covisibility = static_cast<float>(covisibility);
  f.peak_ratio = static_cast<float>(peak_ratio);  // hypothesis ranking carries over
  f.residual_rms_px = static_cast<float>(r.residual_rms);  // photometric RMS
  f.spatial_spread = 1.0f;  // a dense field covers the whole patch
  f.mean_confidence = static_cast<float>(r.mean_confidence);
  f.scale_check = static_cast<float>(scale_check);

  // Metre-scale covariance: (east_px, north_px, yaw) -> (east m, north m, yaw).
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  S(0, 0) = gsd;
  S(1, 1) = gsd;
  Eigen::Matrix3d C = S * r.covariance * S;
  C = addSystematic(C, cov.basemap_bias_sigma_m);
  C = applyFloor(C, cov.floor_position_m, deg2rad(cov.floor_yaw_deg));
  f.covariance = C;
  return f;
}

}  // namespace geoloc
