// T19 unit tests (level 0) for the phase-correlation matcher channel.
//
// Dependency-free on purpose (no gtest / ROS / FFTW): it compiles and runs in a
// bare cross-build container, exactly like geoloc_common's property tests. The
// synthetic images and thresholds mirror the numpy reference implementation
// (which was validated over hundreds of random rotation/translation cases).
//
// Determinism: fixed seed. A non-deterministic test is a broken test.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "geoloc_matcher/phase_corr.hpp"

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

// Blob + sinusoid texture. The blobs give non-periodic structure; the
// sinusoids give periodic structure for the peak_ratio tests (crop-grid-like).
GrayImage makeTexture(int w, int h, uint64_t seed, bool periodic) {
  std::mt19937_64 r(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  GrayImage img(w, h);
  const int nfeat = 60;
  for (int f = 0; f < nfeat; ++f) {
    const double by = uni(r) * h, bx = uni(r) * w;
    const double br = 3.0 + uni(r) * 9.0;
    const double a = uni(r) * 2.0 - 1.0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = x - bx, dy = y - by;
        img.at(x, y) += a * std::exp(-(dx * dx + dy * dy) / (2.0 * br * br));
      }
  }
  if (periodic) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        img.at(x, y) += 0.6 * std::sin(2.0 * kPi * x / 24.0);
        img.at(x, y) += 0.4 * std::cos(2.0 * kPi * y / 19.0);
      }
  }
  double lo = img.data[0], hi = img.data[0];
  for (double v : img.data) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  for (double& v : img.data) v = (v - lo) / (hi - lo);
  return img;
}

// Strongly periodic texture (pure sinusoids, no blobs): the adversarial case
// where phase correlation finds a sharp but period-shifted peak.
GrayImage makePeriodic(int w, int h, double period) {
  GrayImage img(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img.at(x, y) = std::sin(2.0 * kPi * x / period) + std::cos(2.0 * kPi * y / period);
  double lo = img.data[0], hi = img.data[0];
  for (double v : img.data) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  for (double& v : img.data) v = (v - lo) / (hi - lo);
  return img;
}

// Bilinear rotate of a grayscale image, reusing the library's own rotation so
// ground-truth and compensation use the same operator.
GrayImage rotate(const GrayImage& img, double deg) {
  GrayImage out(img.width, img.height);
  phasecorr_detail::bilinearRotate(img.data, img.width, img.height, deg, out.data);
  return out;
}

// Embed `patch` into a textured map at (ox, oy). The map has its own random
// texture elsewhere so the test exercises localisation against a distracting
// background, matching the numpy reference (not a flat field).
GrayImage embed(const GrayImage& patch, int w, int h, uint64_t seed, int ox, int oy) {
  GrayImage ref = makeTexture(w, h, seed, false);
  for (int y = 0; y < patch.height; ++y)
    for (int x = 0; x < patch.width; ++x) ref.at(ox + x, oy + y) = patch.at(x, y);
  return ref;
}

void applyGamma(GrayImage& img, double g) {
  for (double& v : img.data) v = std::pow(std::max(v, 0.0), g);
}

GrayImage resize(GrayImage img, double s) {
  const int w = img.width, h = img.height;
  const int W = static_cast<int>(std::round(w * s)), H = static_cast<int>(std::round(h * s));
  GrayImage out(W, H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const double sx = (x + 0.5) / s - 0.5, sy = (y + 0.5) / s - 0.5;
      const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, w - 2);
      const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, h - 2);
      const double fx = sx - x0, fy = sy - y0;
      out.at(x, y) = img.at(x0, y0) * (1 - fx) * (1 - fy) + img.at(x0 + 1, y0) * fx * (1 - fy) +
                     img.at(x0, y0 + 1) * (1 - fx) * fy + img.at(x0 + 1, y0 + 1) * fx * fy;
    }
  return out;
}

// ---------------------------------------------------------------------------
// T19-U-01: shift recovery. Synthetic shifted pair -> error <= 0.3 px.
// ---------------------------------------------------------------------------
void testShift() {
  std::printf("T19-U-01  shift recovery\n");
  const int ox = 70, oy = 40;  // east = +col, south = +row
  const GrayImage base = makeTexture(128, 128, 1, false);
  const GrayImage ref = embed(base, 256, 256, 2, ox, oy);

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(base, ref);
  check(r.success, "match success");
  checkNear(r.shift_east_px, ox, 0.3, "shift_east_px");
  checkNear(r.shift_north_px, -oy, 0.3, "shift_north_px");
  check(std::abs(r.delta_yaw) < 0.01, "delta_yaw ~ 0");
}

// ---------------------------------------------------------------------------
// T19-U-02: rotation recovery. Synthetic rotated pair -> error <= 0.5 deg.
// ---------------------------------------------------------------------------
void testRotation() {
  std::printf("T19-U-02  rotation recovery\n");
  const double ang_deg = 11.3;
  const GrayImage base = makeTexture(128, 128, 3, false);
  const GrayImage rot = rotate(base, ang_deg);

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(rot, base);
  check(r.success, "match success");
  checkNear(rad2deg(r.delta_yaw), -ang_deg, 0.5, "delta_yaw");
}

// ---------------------------------------------------------------------------
// T19-U-03: illumination robustness. gamma 0.6 vs 1.6 -> shift moves <= 0.5 px.
// ---------------------------------------------------------------------------
void testGamma() {
  std::printf("T19-U-03  illumination (gamma) robustness\n");
  const int ox = 90, oy = 55;
  const GrayImage base = makeTexture(128, 128, 4, false);
  GrayImage query = base;
  applyGamma(query, 0.6);
  GrayImage mapBase = base;
  applyGamma(mapBase, 1.6);
  const GrayImage ref = embed(mapBase, 256, 256, 5, ox, oy);

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(query, ref);
  check(r.success, "match success");
  checkNear(r.shift_east_px, ox, 0.5, "shift_east_px");
  checkNear(r.shift_north_px, -oy, 0.5, "shift_north_px");
}

// ---------------------------------------------------------------------------
// T19-U-04: scale check. Known scale -> no flag; 1.2x -> bad_scale.
// ---------------------------------------------------------------------------
void testScaleCheck() {
  std::printf("T19-U-04  scale check\n");
  const GrayImage base = makeTexture(128, 128, 7, false);
  const GrayImage ref = embed(base, 256, 256, 8, 70, 40);

  PhaseCorrMatcher m;
  const PhaseCorrResult ok = m.match(base, ref);
  check(ok.success, "match success");
  checkNear(ok.scale, 1.0, 0.10, "scale ~ 1");
  check(!ok.bad_scale, "known scale -> no flag");

  const GrayImage scaled = resize(base, 1.2);
  const PhaseCorrResult bad = m.match(scaled, ref);
  check(bad.bad_scale, "1.2x scale -> bad_scale flag");
}

// ---------------------------------------------------------------------------
// T19-U-05: masking. A zero-confidence zone must not change the result.
// ---------------------------------------------------------------------------
void testMasking() {
  std::printf("T19-U-05  confidence masking\n");
  const int ox = 70, oy = 40;
  const GrayImage base = makeTexture(128, 128, 6, false);
  const GrayImage ref = embed(base, 256, 256, 10, ox, oy);

  GrayImage conf(128, 128);
  for (double& v : conf.data) v = 1.0;
  for (int y = 0; y < 30; ++y)
    for (int x = 0; x < 30; ++x) conf.at(x, y) = 0.0;  // zero-confidence corner

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(base, ref, &conf);
  check(r.success, "match success");
  checkNear(r.shift_east_px, ox, 0.3, "shift_east_px");
  checkNear(r.shift_north_px, -oy, 0.3, "shift_north_px");
  // The masked corner must be EXCLUDED from the valid-pixel (covisibility)
  // equivalent, so valid_fraction drops below the fully-confident case.
  check(r.valid_fraction < 1.0, "masked corner reduces covisibility equivalent");
}

// ---------------------------------------------------------------------------
// peak_ratio: good match -> high; periodic structure -> low (the T22 signal).
// ---------------------------------------------------------------------------
void testPeakRatio() {
  std::printf("T19       peak_ratio separates good from periodic\n");
  const GrayImage base = makeTexture(128, 128, 20, false);
  const GrayImage ref = embed(base, 256, 256, 21, 70, 40);
  PhaseCorrMatcher m;
  const PhaseCorrResult good = m.match(base, ref);
  check(good.success, "good match success");
  check(good.peak_ratio > 3.0, "good match has high peak_ratio");

  // A strongly periodic patch in a strongly periodic map: a period-shifted
  // hypothesis is nearly as strong as the true one, so peak_ratio collapses.
  // This is exactly the crop-grid adversarial case from the task card.
  const GrayImage pp = makePeriodic(128, 128, 16.0);
  const GrayImage pr = makePeriodic(256, 256, 16.0);
  const PhaseCorrResult per = m.match(pp, pr);
  check(per.success, "periodic match success");
  check(per.peak_ratio < good.peak_ratio, "periodic peak_ratio is lower");
}

// ---------------------------------------------------------------------------
// Combined rotation + translation, patch embedded in the map.
// ---------------------------------------------------------------------------
void testCombined() {
  std::printf("T19       combined rotation + translation\n");
  const double ang_deg = 7.0;
  const int ox = 80, oy = 60;
  const GrayImage base = makeTexture(128, 128, 9, false);
  const GrayImage rot = rotate(base, ang_deg);
  const GrayImage ref = embed(base, 256, 256, 10, ox, oy);

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(rot, ref);
  check(r.success, "match success");
  checkNear(rad2deg(r.delta_yaw), -ang_deg, 0.5, "delta_yaw");
  checkNear(r.shift_east_px, ox, 0.3, "shift_east_px");
  checkNear(r.shift_north_px, -oy, 0.3, "shift_north_px");
}

// ---------------------------------------------------------------------------
// phaseCorrToFix: pixel shift -> metres via GSD; sign conventions.
// ---------------------------------------------------------------------------
void testToFix() {
  std::printf("T19       SE2Fix field mapping\n");
  const int ox = 70, oy = 40;
  const GrayImage base = makeTexture(128, 128, 11, false);
  const GrayImage ref = embed(base, 256, 256, 12, ox, oy);

  PhaseCorrMatcher m;
  const PhaseCorrResult r = m.match(base, ref);
  const double gsd = 0.5;
  const PhaseCorrFix f = phaseCorrToFix(r, gsd);
  checkNear(f.delta_east, ox * gsd, 0.2, "delta_east = east_px * gsd");
  checkNear(f.delta_north, -oy * gsd, 0.2, "delta_north = north_px * gsd");
  check(f.n_inliers == f.n_correspondences && f.inlier_ratio == 1.0f,
        "sparse-correspondence equivalents");
  check(f.covisibility > 0.0f && f.covisibility <= 1.0f, "covisibility in range");
  check(f.spatial_spread == 1.0f, "global shift -> full spatial spread");
  check(isValidCovariance(f.covariance), "coarse covariance is valid");
}

}  // namespace

int main() {
  testShift();
  testRotation();
  testGamma();
  testScaleCheck();
  testMasking();
  testPeakRatio();
  testCombined();
  testToFix();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
