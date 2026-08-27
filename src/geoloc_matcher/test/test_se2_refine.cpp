// T17 unit tests (level 0) for the SE(2) refinement stage.
//
// Dependency-free on purpose (no gtest / ROS / FFTW): it compiles and runs in a
// bare cross-build container, exactly like geoloc_common's property tests and
// the T19 phase-correlation tests.
//
// Determinism: fixed seed. A non-deterministic test is a broken test.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "geoloc_matcher/se2_refine.hpp"

using namespace geoloc;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* name, const char* detail = "") {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL: %s %s\n", name, detail);
  }
}

void checkNear(double a, double b, double tol, const char* name) {
  ++g_checks;
  if (!(std::abs(a - b) <= tol)) {
    ++g_failures;
    std::printf("  FAIL: %s (got %.6g, want %.6g, tol %.3g)\n", name, a, b, tol);
  }
}

// Structured texture: isotropic blobs (localisation) + sharp rectangles
// (rotational observability). Rotational observability matters: purely
// isotropic blobs make yaw weakly constrained and the refinement cannot pull it
// out of noise, exactly like a real nadir view needs edges/corners to pin yaw.
GrayImage makeTexture(int w, int h, uint64_t seed) {
  std::mt19937_64 r(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  GrayImage img(w, h);
  for (int f = 0; f < 60; ++f) {
    const double by = uni(r) * h, bx = uni(r) * w;
    const double br = 3.0 + uni(r) * 9.0;
    const double a = uni(r) * 2.0 - 1.0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = x - bx, dy = y - by;
        img.at(x, y) += a * std::exp(-(dx * dx + dy * dy) / (2.0 * br * br));
      }
  }
  for (int f = 0; f < 40; ++f) {
    const double bx = uni(r) * w, by = uni(r) * h;
    const double bw = 8.0 + uni(r) * 30.0, bh = 8.0 + uni(r) * 30.0;
    const double a = uni(r) * 2.0 - 1.0;
    const int x0 = std::clamp(static_cast<int>(bx), 0, w - 1);
    const int y0 = std::clamp(static_cast<int>(by), 0, h - 1);
    const int x1 = std::clamp(static_cast<int>(bx + bw), 0, w - 1);
    const int y1 = std::clamp(static_cast<int>(by + bh), 0, h - 1);
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) img.at(x, y) += a;
  }
  double lo = img.data[0], hi = img.data[0];
  for (double v : img.data) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  for (double& v : img.data) v = (v - lo) / (hi - lo);
  return img;
}

// Generate a patch by forward-warping the map through the refiner's OWN warp
// model (patch pixel -> map coordinate). Self-consistent by construction: the
// optimizer and the generator share one definition of the SE(2) warp, so any
// recovered parameter set is the exact global minimum.
GrayImage warpPatch(const GrayImage& map, int pw, int ph, double tx, double ty, double theta) {
  const double cx = (pw - 1) / 2.0, cy = (ph - 1) / 2.0;
  const double ct = std::cos(theta), st = std::sin(theta);
  GrayImage patch(pw, ph);
  for (int y = 0; y < ph; ++y)
    for (int x = 0; x < pw; ++x) {
      const double du = x - cx, dv = y - cy;
      const double u = tx + ct * du - st * dv;
      const double v = ty + st * du + ct * dv;
      double tv = 0.0;
      se2refine_detail::sampleBilinear(map.data, map.width, map.height, u, v, tv);
      patch.at(x, y) = tv;
    }
  return patch;
}

double eastFromTx(double tx, int pw) { return tx - (pw - 1) / 2.0; }
double northFromTy(double ty, int ph) { return (ph - 1) / 2.0 - ty; }

// ---------------------------------------------------------------------------
// T17-refine-U-01: SE(2) recovery from a realistic coarse init (~0.4 px / deg).
// ---------------------------------------------------------------------------
void testRecoverSe2() {
  std::printf("T17-refine-U-01  SE(2) recovery from coarse init\n");
  const GrayImage map = makeTexture(256, 256, 1);
  const int pw = 128, ph = 128;
  const double true_tx = 130.0, true_ty = 120.0, true_theta = -0.09;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  const double e0 = eastFromTx(true_tx, pw), n0 = northFromTy(true_ty, ph), y0 = true_theta;
  const double ce = e0 + 0.4, cn = n0 - 0.3, cy = y0 + 0.4 * kPi / 180.0;

  Se2Refiner r;
  const Se2RefineResult res = r.refine(patch, map, ce, cn, cy);
  check(res.success, "success");
  checkNear(res.shift_east_px, e0, 0.2, "shift_east_px");
  checkNear(res.shift_north_px, n0, 0.2, "shift_north_px");
  checkNear(res.delta_yaw, true_theta, 0.2 * kPi / 180.0, "delta_yaw");
}

// ---------------------------------------------------------------------------
// T17-refine-U-02: sub-pixel accuracy on a fractional translation.
// ---------------------------------------------------------------------------
void testSubpixel() {
  std::printf("T17-refine-U-02  sub-pixel accuracy\n");
  const GrayImage map = makeTexture(256, 256, 3);
  const int pw = 128, ph = 128;
  const double true_tx = 133.37, true_ty = 117.63, true_theta = 0.11;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  const double e0 = eastFromTx(true_tx, pw), n0 = northFromTy(true_ty, ph);
  const double ce = e0 + 0.4, cn = n0 - 0.3, cy = true_theta + 0.003;

  Se2Refiner r;
  const Se2RefineResult res = r.refine(patch, map, ce, cn, cy);
  check(res.success, "success");
  checkNear(res.shift_east_px, e0, 0.2, "shift_east_px");
  checkNear(res.shift_north_px, n0, 0.2, "shift_north_px");
  checkNear(res.delta_yaw, true_theta, 0.2 * kPi / 180.0, "delta_yaw");
}

// ---------------------------------------------------------------------------
// T17-refine-U-03: illumination DC offset (what mean subtraction handles).
// ---------------------------------------------------------------------------
void testIllumination() {
  std::printf("T17-refine-U-03  illumination DC offset\n");
  const GrayImage map = makeTexture(256, 256, 4);
  const int pw = 128, ph = 128;
  const double true_tx = 125.0, true_ty = 122.0, true_theta = -0.05;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  // A global DC offset between patch and map must not bias the result.
  GrayImage map2 = map;
  for (double& v : map2.data) v = v + 0.3;

  const double e0 = eastFromTx(true_tx, pw), n0 = northFromTy(true_ty, ph);
  const double ce = e0 + 1.0, cn = n0 + 1.0, cy = true_theta - 0.004;

  Se2Refiner r;
  const Se2RefineResult res = r.refine(patch, map2, ce, cn, cy);
  check(res.success, "success");
  checkNear(res.shift_east_px, e0, 0.2, "shift_east_px");
  checkNear(res.shift_north_px, n0, 0.2, "shift_north_px");
}

// ---------------------------------------------------------------------------
// T17-refine-U-04: robustness to a masked / occluded region (outlier block).
// ---------------------------------------------------------------------------
void testOcclusion() {
  std::printf("T17-refine-U-04  occlusion robustness\n");
  const GrayImage map = makeTexture(256, 256, 5);
  const int pw = 128, ph = 128;
  const double true_tx = 128.0, true_ty = 128.0, true_theta = 0.0;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  GrayImage occ = patch;
  const GrayImage noise = makeTexture(64, 64, 42);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) occ.at(x, y) = noise.at(x, y);

  const double e0 = eastFromTx(true_tx, pw), n0 = northFromTy(true_ty, ph);
  const double ce = e0 + 0.4, cn = n0 - 0.4, cy = true_theta + 0.003;

  Se2Refiner r;
  const Se2RefineResult res = r.refine(occ, map, ce, cn, cy);
  check(res.success, "success");
  checkNear(res.shift_east_px, e0, 0.4, "shift_east_px");
  checkNear(res.shift_north_px, n0, 0.4, "shift_north_px");
  check(res.inlier_ratio < 1.0, "occluded region drops inlier_ratio");
  check(res.n_inliers < res.n_total, "n_inliers < n_total");
}

// ---------------------------------------------------------------------------
// T17-refine-U-05: honest refusal when the patch maps mostly out of the map.
// ---------------------------------------------------------------------------
void testRefusal() {
  std::printf("T17-refine-U-05  honest refusal on no overlap\n");
  const GrayImage map = makeTexture(256, 256, 6);
  const int pw = 128, ph = 128;
  Se2Refiner r;
  // Patch centre 400 px outside the map: overlap ~0, must refuse (keep coarse).
  const Se2RefineResult res =
      r.refine(warpPatch(map, pw, ph, 128, 128, 0.0), map, 400.0, 0.0, 0.0);
  check(!res.success, "no overlap -> success=false");
}

// ---------------------------------------------------------------------------
// T17-refine-U-06: end-to-end -- phase correlation coarse -> refined accurate.
// ---------------------------------------------------------------------------
void testEndToEnd() {
  std::printf("T17-refine-U-06  phase corr coarse -> refined\n");
  const double ang_deg = 6.0;
  const int ox = 80, oy = 60;
  const GrayImage base = makeTexture(128, 128, 9);
  std::vector<double> rot;
  phasecorr_detail::bilinearRotate(base.data, base.width, base.height, ang_deg, rot);
  GrayImage patch = base;
  patch.data = std::move(rot);
  GrayImage map = makeTexture(256, 256, 10);
  for (int y = 0; y < base.height; ++y)
    for (int x = 0; x < base.width; ++x) map.at(ox + x, oy + y) = base.at(x, y);

  const double e0 = ox, n0 = -oy, y0 = -ang_deg * kPi / 180.0;

  PhaseCorrMatcher pc;
  const PhaseCorrResult coarse = pc.match(patch, map);
  check(coarse.success, "phase corr success");
  checkNear(coarse.shift_east_px, e0, 0.5, "coarse shift_east");
  checkNear(coarse.shift_north_px, n0, 0.5, "coarse shift_north");

  Se2Refiner r;
  const Se2RefineResult res =
      r.refine(patch, map, coarse.shift_east_px, coarse.shift_north_px, coarse.delta_yaw);
  check(res.success, "refine success");
  checkNear(res.shift_east_px, e0, 0.2, "refined shift_east");
  checkNear(res.shift_north_px, n0, 0.2, "refined shift_north");
  checkNear(res.delta_yaw, y0, 0.2 * kPi / 180.0, "refined yaw");
}

// ---------------------------------------------------------------------------
// T17-refine-U-07: stress -- larger coarse error still improves and stays sane.
// ---------------------------------------------------------------------------
void testStress() {
  std::printf("T17-refine-U-07  larger coarse error (robustness)\n");
  const GrayImage map = makeTexture(256, 256, 12);
  const int pw = 128, ph = 128;
  const double true_tx = 128.0, true_ty = 128.0, true_theta = 0.0;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  const double e0 = eastFromTx(true_tx, pw), n0 = northFromTy(true_ty, ph);
  const double ce = e0 + 2.0, cn = n0 - 1.5, cy = true_theta + 0.5 * kPi / 180.0;

  Se2Refiner r;
  const Se2RefineResult res = r.refine(patch, map, ce, cn, cy);
  check(res.success, "success");
  // At minimum the refinement must move closer to the truth than the coarse init
  // (never wander away), and all fields must stay finite with a valid covariance.
  const double err0 = std::hypot(ce - e0, cn - n0);
  const double err1 = std::hypot(res.shift_east_px - e0, res.shift_north_px - n0);
  check(err1 < err0, "refined pose closer than coarse");
  check(std::isfinite(res.residual_rms) && std::isfinite(res.delta_yaw), "finite fields");
  check(isValidCovariance(res.covariance), "covariance valid");
}

// ---------------------------------------------------------------------------
// se2RefineToFix: metres mapping, sign conventions, valid covariance.
// ---------------------------------------------------------------------------
void testToFix() {
  std::printf("T17-refine       SE2Fix field mapping\n");
  const GrayImage map = makeTexture(256, 256, 11);
  const int pw = 128, ph = 128;
  const double true_tx = 120.0, true_ty = 130.0, true_theta = -0.07;
  const GrayImage patch = warpPatch(map, pw, ph, true_tx, true_ty, true_theta);

  Se2Refiner r;
  const Se2RefineResult res = r.refine(patch, map, eastFromTx(true_tx, pw) + 0.4,
                                       northFromTy(true_ty, ph) - 0.4, true_theta + 0.003);
  check(res.success, "success");

  const double gsd = 0.5;
  const PhaseCorrFix f = se2RefineToFix(res, gsd, 2.1, 1.0, 0.9);
  checkNear(f.delta_east, res.shift_east_px * gsd, 1e-9, "delta_east = east_px * gsd");
  checkNear(f.delta_north, res.shift_north_px * gsd, 1e-9, "delta_north = north_px * gsd");
  check(std::abs(f.delta_yaw) <= kPi, "delta_yaw normalised");
  check(f.n_inliers > 0, "n_inliers > 0");
  check(f.inlier_ratio > 0.0f && f.inlier_ratio <= 1.0f, "inlier_ratio in range");
  check(isValidCovariance(f.covariance), "refined covariance is valid");
}

}  // namespace

int main() {
  testRecoverSe2();
  testSubpixel();
  testIllumination();
  testOcclusion();
  testRefusal();
  testEndToEnd();
  testStress();
  testToFix();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
