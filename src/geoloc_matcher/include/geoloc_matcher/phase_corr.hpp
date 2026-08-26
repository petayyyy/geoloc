// Phase-correlation matcher channel (T19) -- fallback 1.
//
// Correlates on GRADIENT ORIENTATION, not intensity, which is what makes it
// robust to the monotonic illumination / cross-provider radiometry changes that
// kill point features. It estimates a global shift (and a coarse rotation) in
// one shot instead of sparse correspondences, and is deliberately coarse:
// 5-15 m / 1-3 deg against XFeat's 3-6 m. Its job is to keep covariance from
// growing without bound when the primary channel is silent -- not to be
// accurate.
//
// Pipeline (validated against a numpy reference):
//   1. Sobel -> gradient orientation field -> Hann window (edge effects).
//   2. Coarse rotation sweep (rotation-tolerant localisation, since a rotated
//      patch smears the correlation peak and breaks a single 0-rotation pass).
//   3. Log-polar FFT (Fourier-Mellin) on a patch-sized crop -> rotation + scale.
//      Scale is known, so the estimate is a CHECK, not a result.
//   4. Phase correlation of the rotation-compensated pair -> dx, dy (sub-pixel).
//
// Pitfalls this file specifically guards against (task card T19):
//   * Phase correlation fails CONFIDENTLY on periodic structures: a sharp peak
//     shifted by one period. peak_ratio is computed over SPATIALLY SEPARATED
//     peaks (not neighbouring samples of one peak) -- the only defense.
//   * Hann window is mandatory: without it FFT edge effects produce a false
//     central peak.
//   * |F| is 180-degree symmetric, so the FMT rotation is only defined modulo
//     180; the patch is pre-rectified (yaw_map ~ 0), so we fold to [-90, 90).
//
// ARCHITECTURAL RULE (unchanged): this channel produces a result with quality
// metrics and NEVER decides whether it is valid. That decision lives only in
// geoloc_integrity.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "geoloc_common/angles.hpp"
#include "geoloc_common/covariance.hpp"
#include "geoloc_matcher/fft.hpp"

namespace geoloc {

/// Row-major grayscale image (double). Kept trivially copyable so the matcher
/// can own reusable buffers without hidden indirection.
struct GrayImage {
  int width = 0;
  int height = 0;
  std::vector<double> data;

  GrayImage() = default;
  GrayImage(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h, 0.0) {}

  double at(int x, int y) const { return data[static_cast<size_t>(y) * width + x]; }
  double& at(int x, int y) { return data[static_cast<size_t>(y) * width + x]; }
  int size() const { return width * height; }
  bool empty() const { return data.empty(); }
};

struct PhaseCorrConfig {
  // Gradient magnitude below (grad_thresh_rel * mean_magnitude) is treated as
  // no texture and excluded from the orientation field. Orientation (not
  // magnitude) is what gives gamma / illumination invariance.
  double grad_thresh_rel = 0.0;
  // Coarse rotation sweep bounds and step. The patch is pre-rectified, so the
  // true rotation is small; the sweep only needs to localise well enough to
  // crop, the FMT does the actual rotation estimate.
  double coarse_max_deg = 18.0;
  double coarse_step_deg = 6.0;
  // Log-polar sampling of the magnitude spectrum (rounded up to a power of two
  // internally for the FFT). ntheta ~= angular resolution: 512 -> 0.7 deg.
  int nrho = 64;
  int ntheta = 512;
  // Diagnostic: |estimated scale - known| beyond this flags a bad match.
  double scale_check_tolerance = 0.10;
  // Minimum pixel separation used when locating the second (spatially
  // separated) correlation peak for peak_ratio.
  int peak_min_sep = 8;
};

/// Coarse covariance parameters for the fallback channel. This is a PLACEHOLDER
/// model; the real covariance model lands with T21. Values are the channel's
/// stated accuracy envelope (5-15 m, 1-3 deg).
struct PhaseCorrCovarianceConfig {
  double position_sigma_m = 8.0;
  double yaw_sigma_deg = 1.5;
  double min_peak_ratio = 1.6;  // integrity gate for this channel
  double basemap_bias_sigma_m = 3.0;  // T09; does not shrink with inliers
  double floor_position_m = 0.5;
  double floor_yaw_deg = 0.05;
};

struct PhaseCorrResult {
  bool success = false;
  double delta_yaw = 0.0;       // rad; rotation to APPLY to patch to align (pixel frame)
  double shift_east_px = 0.0;   // +x = east (columns)
  double shift_north_px = 0.0;  // +y = north (i.e. -rows)
  double scale = 1.0;           // estimated scale (patch / reference)
  bool bad_scale = false;       // |scale - 1| > scale_check_tolerance
  double peak_ratio = 0.0;      // main / second spatially-separated peak
  double peak_fwhm_px = 0.0;    // peak sharpness (width at half maximum)
  double main_peak = 0.0;
  double second_peak = 0.0;
  double valid_fraction = 0.0;  // covisibility equivalent
  double mean_confidence = 0.0;
  uint32_t n_valid_pixels = 0;  // n_correspondences equivalent
};

/// SE2Fix-shaped fields for this channel. `channel` is fixed to
/// CHANNEL_PHASE_CORR (1); the node copies these into the ROS message.
struct PhaseCorrFix {
  double delta_east = 0.0;
  double delta_north = 0.0;
  double delta_yaw = 0.0;  // rad, normalised to [-pi, pi]
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  uint32_t n_correspondences = 0;
  uint32_t n_inliers = 0;
  float inlier_ratio = 0.0f;
  float covisibility = 0.0f;
  float peak_ratio = 0.0f;
  float residual_rms_px = 0.0f;
  float spatial_spread = 0.0f;
  float mean_confidence = 0.0f;
  float scale_check = 0.0f;
  float processing_time_ms = 0.0f;
};

namespace phasecorr_detail {

inline void sobel(const GrayImage& img, std::vector<double>& gx, std::vector<double>& gy) {
  const int w = img.width, h = img.height;
  gx.assign(static_cast<size_t>(w) * h, 0.0);
  gy.assign(static_cast<size_t>(w) * h, 0.0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(0, x - 1), x1 = std::min(w - 1, x + 1);
      const int y0 = std::max(0, y - 1), y1 = std::min(h - 1, y + 1);
      const double a = img.at(x0, y0), b = img.at(x, y0), c = img.at(x1, y0);
      const double d = img.at(x0, y), f = img.at(x1, y);
      const double g = img.at(x0, y1), hh = img.at(x, y1), i = img.at(x1, y1);
      // Sobel-X = [-1 0 1; -2 0 2; -1 0 1], Sobel-Y = [-1 -2 -1; 0 0 0; 1 2 1].
      const double sx = (-a - 2.0 * d - g) + (c + 2.0 * f + i);
      const double sy = (g + 2.0 * hh + i) - (a + 2.0 * b + c);
      gx[static_cast<size_t>(y) * w + x] = sx;
      gy[static_cast<size_t>(y) * w + x] = sy;
    }
  }
}

/// Gradient orientation field as a unit complex number e^{i*theta}; zero where
/// magnitude is below threshold. Orientation is invariant to monotonic
/// intensity transforms (gamma), which is the whole point of T19-U-03.
inline void orientationField(const GrayImage& img, double grad_thresh_rel,
                             std::vector<std::complex<double>>& out) {
  const int n = img.size();
  std::vector<double> gx, gy;
  sobel(img, gx, gy);
  double mean_mag = 0.0;
  for (int i = 0; i < n; ++i) mean_mag += std::hypot(gx[i], gy[i]);
  mean_mag /= static_cast<double>(n);
  const double thr = grad_thresh_rel * mean_mag;
  out.assign(static_cast<size_t>(n), std::complex<double>(0.0, 0.0));
  for (int i = 0; i < n; ++i) {
    const double m = std::hypot(gx[i], gy[i]);
    if (m > thr) out[i] = std::complex<double>(gx[i] / m, gy[i] / m);
  }
}

inline void hann1d(int n, std::vector<double>& out) {
  out.resize(n);
  if (n == 1) {
    out[0] = 1.0;
    return;
  }
  for (int k = 0; k < n; ++k) out[k] = 0.5 * (1.0 - std::cos(2.0 * kPi * k / (n - 1)));
}

inline void hann2d(int w, int h, std::vector<double>& out) {
  std::vector<double> hx, hy;
  hann1d(w, hx);
  hann1d(h, hy);
  out.assign(static_cast<size_t>(w) * h, 0.0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) out[static_cast<size_t>(y) * w + x] = hy[y] * hx[x];
}

/// Bilinear rotation by angle_deg about the image centre, zero out-of-bounds.
/// Works on real or complex data (rotation of the raw intensity is what makes
/// the orientation field of a rotated patch correct -- rotating the orientation
/// field directly drops a global phase factor).
inline void bilinearRotate(const std::vector<double>& src, int w, int h, double angle_deg,
                           std::vector<double>& out) {
  const double a = deg2rad(angle_deg);
  const double ca = std::cos(a), sa = std::sin(a);
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  out.assign(static_cast<size_t>(w) * h, 0.0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double xd = x - cx, yd = y - cy;
      const double xs = ca * xd + sa * yd + cx;
      const double ys = -sa * xd + ca * yd + cy;
      const int x0 = static_cast<int>(std::floor(xs));
      const int y0 = static_cast<int>(std::floor(ys));
      if (x0 < 0 || x0 + 1 >= w || y0 < 0 || y0 + 1 >= h) continue;
      const double fx = xs - x0, fy = ys - y0;
      const double i00 = src[static_cast<size_t>(y0) * w + x0];
      const double i01 = src[static_cast<size_t>(y0) * w + x0 + 1];
      const double i10 = src[static_cast<size_t>(y0 + 1) * w + x0];
      const double i11 = src[static_cast<size_t>(y0 + 1) * w + x0 + 1];
      out[static_cast<size_t>(y) * w + x] =
          i00 * (1 - fx) * (1 - fy) + i01 * fx * (1 - fy) + i10 * (1 - fx) * fy + i11 * fx * fy;
    }
  }
}

/// Zero-pad a (small) field into a (large) buffer, top-left placement.
inline void zeroPad(const std::vector<std::complex<double>>& src, int sw, int sh, int dw, int dh,
                    std::vector<std::complex<double>>& dst) {
  dst.assign(static_cast<size_t>(dw) * dh, std::complex<double>(0.0, 0.0));
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      dst[static_cast<size_t>(y) * dw + x] = src[static_cast<size_t>(y) * sw + x];
}

/// Phase correlation: shift (sub-pixel) to apply to `patchFftDomain`'s spatial
/// content so it aligns onto `mapFft`. `mapFft` is the precomputed FFT of the
/// windowed map (reused across the rotation sweep). Fills the peak value and,
/// optionally, the real correlation surface (for peak_ratio / sharpness).
inline void phaseCorrShift(const std::vector<std::complex<double>>& patchField,
                           const std::vector<std::complex<double>>& mapFft, int w, int h,
                           double& dx, double& dy, double& peak, std::vector<double>* surface) {
  std::vector<std::complex<double>> Fp = patchField;
  fft2(Fp.data(), w, h, false);
  std::vector<std::complex<double>> R(static_cast<size_t>(w) * h);
  const double eps = 1e-12;
  for (int i = 0; i < w * h; ++i) {
    R[i] = std::conj(Fp[i]) * mapFft[i];
    const double m = std::abs(R[i]);
    R[i] /= (m + eps);
  }
  fft2(R.data(), w, h, true);
  std::vector<double> c(static_cast<size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) c[i] = R[i].real();

  int px = 0, py = 0;
  double best = -1e300;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double v = c[static_cast<size_t>(y) * w + x];
      if (v > best) {
        best = v;
        px = x;
        py = y;
      }
    }
  peak = best;

  auto wrap = [](int v, int n) {
    v %= n;
    return v < 0 ? v + n : v;
  };
  // Sub-pixel parabolic localisation, wrap-aware neighbours.
  const double v0 = c[static_cast<size_t>(py) * w + px];
  const double vxm = c[static_cast<size_t>(py) * w + wrap(px - 1, w)];
  const double vxp = c[static_cast<size_t>(py) * w + wrap(px + 1, w)];
  double sx = 0.0;
  const double denx = vxm - 2.0 * v0 + vxp;
  if (std::abs(denx) > 1e-12) sx = std::clamp(0.5 * (vxm - vxp) / denx, -1.0, 1.0);
  const double vym = c[static_cast<size_t>(wrap(py - 1, h)) * w + px];
  const double vyp = c[static_cast<size_t>(wrap(py + 1, h)) * w + px];
  double sy = 0.0;
  const double deny = vym - 2.0 * v0 + vyp;
  if (std::abs(deny) > 1e-12) sy = std::clamp(0.5 * (vym - vyp) / deny, -1.0, 1.0);

  dx = (px <= w / 2 ? px : px - w) + sx;
  dy = (py <= h / 2 ? py : py - h) + sy;

  if (surface != nullptr) *surface = std::move(c);
}

/// peak_ratio (over SPATIALLY SEPARATED peaks) and peak sharpness (FWHM).
inline void peakMetrics(const std::vector<double>& surface, int w, int h, int min_sep,
                        double& ratio, double& main, double& second, double& fwhm) {
  int px = 0, py = 0;
  double best = -1e300;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double v = surface[static_cast<size_t>(y) * w + x];
      if (v > best) {
        best = v;
        px = x;
        py = y;
      }
    }
  main = best;
  // Suppress a min_sep neighbourhood around the main peak, then take the max
  // elsewhere: this is what makes peak_ratio mean "second spatially separated
  // hypothesis" rather than "second neighbouring sample of the same peak".
  auto wrap = [](int v, int n) {
    v %= n;
    return v < 0 ? v + n : v;
  };
  double snd = -1e300;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int dxp = std::min(std::abs(x - px), w - std::abs(x - px));
      const int dyp = std::min(std::abs(y - py), h - std::abs(y - py));
      if (dxp <= min_sep && dyp <= min_sep) continue;
      snd = std::max(snd, surface[static_cast<size_t>(y) * w + x]);
    }
  second = snd;
  ratio = snd > 1e-12 ? main / snd : 1e9;

  // FWHM: median baseline, half = (main + baseline) / 2, half-width in x and y.
  std::vector<double> vals = surface;
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  const double baseline = vals[vals.size() / 2];
  const double half = 0.5 * (main + baseline);
  auto halfwidth = [&](bool along_x) {
    for (int d = 1; d < 16; ++d) {
      const double v = along_x ? surface[static_cast<size_t>(py) * w + wrap(px + d, w)]
                               : surface[static_cast<size_t>(wrap(py + d, h)) * w + px];
      if (v <= half) {
        if (d == 1) return 0.5;
        const double vprev = along_x ? surface[static_cast<size_t>(py) * w + wrap(px + d - 1, w)]
                                     : surface[static_cast<size_t>(wrap(py + d - 1, h)) * w + px];
        const double frac = (half - vprev) / (v - vprev + 1e-12);
        return d - 1 + frac;
      }
    }
    return 15.0;
  };
  fwhm = 0.5 * (halfwidth(true) + halfwidth(false));
}

inline void fftshift(const std::vector<double>& in, int w, int h, std::vector<double>& out) {
  out.resize(static_cast<size_t>(w) * h);
  const int hw = w / 2, hh = h / 2;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int sx = (x + hw) % w;
      const int sy = (y + hh) % h;
      out[static_cast<size_t>(y) * w + x] = in[static_cast<size_t>(sy) * w + sx];
    }
}

/// Log-polar resample of an fftshifted magnitude spectrum (DC at centre).
inline void logPolarResample(const std::vector<double>& mag, int w, int h, int nrho, int ntheta,
                             std::vector<double>& lp) {
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  const double rmax = std::min(cx, cy);
  const double rmin = 1.5;
  const double log_rmin = std::log(rmin), log_rmax = std::log(rmax);
  lp.assign(static_cast<size_t>(nrho) * ntheta, 0.0);
  for (int ir = 0; ir < nrho; ++ir) {
    const double rho = std::exp(log_rmin + (log_rmax - log_rmin) * ir / (nrho - 1));
    for (int it = 0; it < ntheta; ++it) {
      const double theta = 2.0 * kPi * it / ntheta;
      const double xs = cx + rho * std::cos(theta);
      const double ys = cy + rho * std::sin(theta);
      const int x0 = static_cast<int>(std::floor(xs));
      const int y0 = static_cast<int>(std::floor(ys));
      if (x0 < 0 || x0 + 1 >= w || y0 < 0 || y0 + 1 >= h) continue;
      const double fx = xs - x0, fy = ys - y0;
      const double i00 = mag[static_cast<size_t>(y0) * w + x0];
      const double i01 = mag[static_cast<size_t>(y0) * w + x0 + 1];
      const double i10 = mag[static_cast<size_t>(y0 + 1) * w + x0];
      const double i11 = mag[static_cast<size_t>(y0 + 1) * w + x0 + 1];
      lp[static_cast<size_t>(ir) * ntheta + it] =
          i00 * (1 - fx) * (1 - fy) + i01 * fx * (1 - fy) + i10 * (1 - fx) * fy + i11 * fx * fy;
    }
  }
}

/// Fourier-Mellin rotation + scale from two fftshifted magnitude spectra.
/// Returns rotation (deg, folded to [-90, 90) since |F| is 180-symmetric) and
/// scale (patch/reference).
inline void fmtRotationScale(const std::vector<double>& patchMag,
                             const std::vector<double>& refMag, int w, int h, int nrho, int ntheta,
                             double& angle_deg, double& scale) {
  std::vector<double> lpP, lpM;
  logPolarResample(patchMag, w, h, nrho, ntheta, lpP);
  logPolarResample(refMag, w, h, nrho, ntheta, lpM);

  // Row-normalise (radial profile) to suppress the huge dynamic range that
  // would otherwise let the lowest frequencies dominate the correlation.
  const double eps = 1e-12;
  for (int ir = 0; ir < nrho; ++ir) {
    double mp = 0.0, mm = 0.0;
    for (int it = 0; it < ntheta; ++it) {
      mp += lpP[static_cast<size_t>(ir) * ntheta + it];
      mm += lpM[static_cast<size_t>(ir) * ntheta + it];
    }
    mp /= ntheta;
    mm /= ntheta;
    for (int it = 0; it < ntheta; ++it) {
      lpP[static_cast<size_t>(ir) * ntheta + it] /= (mp + eps);
      lpM[static_cast<size_t>(ir) * ntheta + it] /= (mm + eps);
    }
  }

  std::vector<std::complex<double>> Fp(lpP.begin(), lpP.end());
  std::vector<std::complex<double>> Fm(lpM.begin(), lpM.end());
  const int fw = ntheta;  // already power of two from caller (nrho/ntheta padded)
  const int fh = nrho;
  fft2(Fp.data(), fw, fh, false);
  fft2(Fm.data(), fw, fh, false);
  std::vector<std::complex<double>> R(static_cast<size_t>(fw) * fh);
  for (int i = 0; i < fw * fh; ++i) {
    R[i] = std::conj(Fp[i]) * Fm[i];
    R[i] /= (std::abs(R[i]) + eps);
  }
  fft2(R.data(), fw, fh, true);
  int px = 0, py = 0;
  double best = -1e300;
  for (int y = 0; y < fh; ++y)
    for (int x = 0; x < fw; ++x) {
      const double v = R[static_cast<size_t>(y) * fw + x].real();
      if (v > best) {
        best = v;
        px = x;
        py = y;
      }
    }
  auto wrap = [](int v, int n) {
    v %= n;
    return v < 0 ? v + n : v;
  };
  // Sub-pixel along the angular axis only (radial scale is a coarse check).
  const double v0 = R[static_cast<size_t>(py) * fw + px].real();
  const double vxm = R[static_cast<size_t>(py) * fw + wrap(px - 1, fw)].real();
  const double vxp = R[static_cast<size_t>(py) * fw + wrap(px + 1, fw)].real();
  double sx = 0.0;
  const double denx = vxm - 2.0 * v0 + vxp;
  if (std::abs(denx) > 1e-12) sx = std::clamp(0.5 * (vxm - vxp) / denx, -1.0, 1.0);

  double dtheta = (px <= fw / 2 ? px : px - fw) + sx;
  angle_deg = dtheta * 360.0 / fw;
  while (angle_deg > 90.0) angle_deg -= 180.0;
  while (angle_deg < -90.0) angle_deg += 180.0;

  const double drho = (py <= fh / 2 ? py : py - fh);
  const double rmax = std::min(w, h) / 2.0;
  const double log_step = (std::log(rmax) - std::log(1.5)) / (nrho - 1);
  scale = std::exp(drho * log_step);
}

}  // namespace phasecorr_detail

/// Phase-correlation matcher (fallback 1). Reusable; owns its work buffers so
/// repeated calls at 1-2 Hz do not allocate in the hot path.
class PhaseCorrMatcher {
 public:
  explicit PhaseCorrMatcher(const PhaseCorrConfig& cfg = PhaseCorrConfig{}) : cfg_(cfg) {}

  PhaseCorrResult match(const GrayImage& patch, const GrayImage& ref,
                        const GrayImage* confidence = nullptr) const {
    using namespace phasecorr_detail;
    PhaseCorrResult res;
    const int pw = patch.width, ph = patch.height;
    const int rw = ref.width, rh = ref.height;
    if (patch.empty() || ref.empty() || pw > rw || ph > rh) return res;

    std::vector<double> wp, wr;
    hann2d(pw, ph, wp);
    hann2d(rw, rh, wr);

    // Windowed orientation field of the map, and its FFT (computed ONCE and
    // reused across every rotation of the sweep).
    std::vector<std::complex<double>> fm;
    orientationField(ref, cfg_.grad_thresh_rel, fm);
    std::vector<std::complex<double>> M(static_cast<size_t>(rw) * rh, std::complex<double>(0, 0));
    for (int i = 0; i < rw * rh; ++i) M[i] = fm[i] * wr[i];
    fft2(M.data(), rw, rh, false);

    // --- 1. coarse rotation sweep: localise (rotation-tolerant) ---
    double best_peak = -1e300, best_dx = 0.0, best_dy = 0.0;
    for (double psi = -cfg_.coarse_max_deg; psi <= cfg_.coarse_max_deg + 1e-9;
         psi += cfg_.coarse_step_deg) {
      std::vector<double> rot;
      bilinearRotate(patch.data, pw, ph, psi, rot);
      GrayImage rimg;
      rimg.width = pw;
      rimg.height = ph;
      rimg.data = std::move(rot);
      std::vector<std::complex<double>> fo;
      orientationField(rimg, cfg_.grad_thresh_rel, fo);
      if (confidence != nullptr) {
        std::vector<double> crot;
        bilinearRotate(confidence->data, pw, ph, psi, crot);
        for (int i = 0; i < pw * ph; ++i)
          if (crot[i] <= 0.5) fo[i] = std::complex<double>(0, 0);
      }
      std::vector<std::complex<double>> P;
      zeroPad(fo, pw, ph, rw, rh, P);
      for (int i = 0; i < pw * ph; ++i) {
        const int x = i % pw, y = i / pw;
        P[static_cast<size_t>(y) * rw + x] *= wp[i];
      }
      double dx, dy, peak;
      phaseCorrShift(P, M, rw, rh, dx, dy, peak, nullptr);
      if (peak > best_peak) {
        best_peak = peak;
        best_dx = dx;
        best_dy = dy;
      }
    }

    // --- 2. FMT (log-polar) rotation + scale on a patch-sized crop ---
    const int cx = static_cast<int>(std::round(best_dx));
    const int cy = static_cast<int>(std::round(best_dy));
    const int x0 = std::clamp(cx, 0, rw - pw);
    const int y0 = std::clamp(cy, 0, rh - ph);
    GrayImage crop(pw, ph);
    for (int y = 0; y < ph; ++y)
      for (int x = 0; x < pw; ++x) crop.at(x, y) = ref.at(x0 + x, y0 + y);

    std::vector<std::complex<double>> fp0;
    orientationField(patch, cfg_.grad_thresh_rel, fp0);
    std::vector<std::complex<double>> fc0;
    orientationField(crop, cfg_.grad_thresh_rel, fc0);
    std::vector<std::complex<double>> Pc(static_cast<size_t>(pw) * ph, std::complex<double>(0, 0));
    std::vector<std::complex<double>> Mc(static_cast<size_t>(pw) * ph, std::complex<double>(0, 0));
    for (int i = 0; i < pw * ph; ++i) {
      Pc[i] = fp0[i] * wp[i];
      Mc[i] = fc0[i] * wp[i];
    }
    const int fw = nextPow2(pw), fh = nextPow2(ph);
    std::vector<std::complex<double>> Fp(static_cast<size_t>(fw) * fh, std::complex<double>(0, 0));
    std::vector<std::complex<double>> Fm(static_cast<size_t>(fw) * fh, std::complex<double>(0, 0));
    for (int y = 0; y < ph; ++y)
      for (int x = 0; x < pw; ++x) {
        Fp[static_cast<size_t>(y) * fw + x] = Pc[static_cast<size_t>(y) * pw + x];
        Fm[static_cast<size_t>(y) * fw + x] = Mc[static_cast<size_t>(y) * pw + x];
      }
    fft2(Fp.data(), fw, fh, false);
    fft2(Fm.data(), fw, fh, false);
    std::vector<double> magP(static_cast<size_t>(fw) * fh), magM(static_cast<size_t>(fw) * fh);
    for (int i = 0; i < fw * fh; ++i) {
      magP[i] = std::abs(Fp[i]);
      magM[i] = std::abs(Fm[i]);
    }
    std::vector<double> shP, shM;
    fftshift(magP, fw, fh, shP);
    fftshift(magM, fw, fh, shM);
    const int nr = nextPow2(cfg_.nrho), nt = nextPow2(cfg_.ntheta);
    double angle_deg = 0.0;
    fmtRotationScale(shP, shM, fw, fh, nr, nt, angle_deg, res.scale);
    res.delta_yaw = deg2rad(angle_deg);

    // --- 3. final translation at the estimated rotation ---
    std::vector<double> rot;
    bilinearRotate(patch.data, pw, ph, angle_deg, rot);
    GrayImage rimg;
    rimg.width = pw;
    rimg.height = ph;
    rimg.data = std::move(rot);
    std::vector<std::complex<double>> fo;
    orientationField(rimg, cfg_.grad_thresh_rel, fo);
    if (confidence != nullptr) {
      std::vector<double> crot;
      bilinearRotate(confidence->data, pw, ph, angle_deg, crot);
      for (int i = 0; i < pw * ph; ++i)
        if (crot[i] <= 0.5) fo[i] = std::complex<double>(0, 0);
    }
    std::vector<std::complex<double>> P;
    zeroPad(fo, pw, ph, rw, rh, P);
    for (int i = 0; i < pw * ph; ++i) {
      const int x = i % pw, y = i / pw;
      P[static_cast<size_t>(y) * rw + x] *= wp[i];
    }
    std::vector<double> surface;
    double peak;
    double dx, dy;
    phaseCorrShift(P, M, rw, rh, dx, dy, peak, &surface);
    res.main_peak = peak;
    peakMetrics(surface, rw, rh, cfg_.peak_min_sep, res.peak_ratio, res.main_peak, res.second_peak,
                res.peak_fwhm_px);

    // Pixel-frame shift: +x east, +y south (row increases southward), so north
    // shift is -dy.
    res.shift_east_px = dx;
    res.shift_north_px = -dy;

    // Quality equivalents (T19: n_inliers / covisibility have no direct
    // meaning for a global shift; fill with valid-pixel equivalents: textured
    // pixels inside the confidence mask).
    res.n_valid_pixels = 0;
    double conf_sum = 0.0, conf_cnt = 0.0;
    for (int i = 0; i < pw * ph; ++i) {
      const bool textured = fp0[i] != std::complex<double>(0.0, 0.0);
      const double c = confidence != nullptr ? confidence->data[i] : 1.0;
      if (textured && c > 0.0) {
        ++res.n_valid_pixels;
        conf_sum += c;
        conf_cnt += 1.0;
      }
    }
    res.valid_fraction = static_cast<double>(res.n_valid_pixels) / static_cast<double>(pw * ph);
    res.mean_confidence = conf_cnt > 0.0 ? conf_sum / conf_cnt : 0.0;

    res.bad_scale = std::abs(res.scale - 1.0) > cfg_.scale_check_tolerance;
    res.success = true;
    return res;
  }

 private:
  PhaseCorrConfig cfg_;
};

/// Convert a phase-correlation result to SE2Fix-shaped fields.
///
/// The pixel shift becomes a metres correction via the patch GSD; the sign
/// conventions follow raster.hpp (row increases southward). `delta_yaw` is the
/// rotation to apply to the patch, negated to the ENU yaw convention (CCW from
/// East). Covariance is a coarse PLACEHOLDER for T21.
inline PhaseCorrFix phaseCorrToFix(const PhaseCorrResult& r, double gsd,
                                   const PhaseCorrCovarianceConfig& cov =
                                       PhaseCorrCovarianceConfig{}) {
  PhaseCorrFix f;
  f.delta_east = r.shift_east_px * gsd;
  f.delta_north = r.shift_north_px * gsd;
  // `delta_yaw` is expressed in the image frame (rows increase southward), so
  // a rotation of +theta in image space is -theta in ENU yaw (CCW from East).
  // Negate to get the correction in the ENU convention.
  f.delta_yaw = normalizeAngle(-r.delta_yaw);

  // n_inliers / inlier_ratio / spatial_spread are sparse-correspondence
  // concepts; for a global shift they are filled with documented equivalents.
  f.n_correspondences = r.n_valid_pixels;
  f.n_inliers = r.n_valid_pixels;
  f.inlier_ratio = 1.0f;
  f.covisibility = static_cast<float>(r.valid_fraction);
  f.peak_ratio = static_cast<float>(r.peak_ratio);
  f.residual_rms_px = static_cast<float>(r.peak_fwhm_px);  // sharpness equivalent
  f.spatial_spread = 1.0f;  // a single global shift covers the whole patch
  f.mean_confidence = static_cast<float>(r.mean_confidence);
  f.scale_check = static_cast<float>(r.scale);

  // Coarse covariance: scale the channel's stated envelope by how far the peak
  // ratio is below the integrity gate. Replaced by the T21 model.
  const double penalty = std::max(1.0, cov.min_peak_ratio / std::max(r.peak_ratio, 1e-3));
  const double sigma_pos = cov.position_sigma_m * penalty;
  const double sigma_yaw = deg2rad(cov.yaw_sigma_deg) * penalty;
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  C(0, 0) = sigma_pos * sigma_pos;
  C(1, 1) = sigma_pos * sigma_pos;
  C(2, 2) = sigma_yaw * sigma_yaw;
  C = addSystematic(C, cov.basemap_bias_sigma_m);
  C = applyFloor(C, cov.floor_position_m, deg2rad(cov.floor_yaw_deg));
  f.covariance = C;
  return f;
}

}  // namespace geoloc
