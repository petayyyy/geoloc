// Property tests for geoloc_common, per task card T01 (T01-U-01 .. T01-U-05).
//
// Deliberately dependency-free (no gtest) so it compiles and runs anywhere,
// including a bare cross-build container. In the ROS workspace this is wrapped
// by ament_add_test; the assertions are the same either way.
//
// Determinism: fixed seed. A non-deterministic test is a broken test.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "geoloc_common/angles.hpp"
#include "geoloc_common/covariance.hpp"
#include "geoloc_common/geodetic.hpp"
#include "geoloc_common/raster.hpp"
#include "geoloc_common/se2.hpp"

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
    std::printf("  FAIL: %s (got %.17g, want %.17g, tol %.3g, diff %.3g)\n", name, a, b, tol,
                std::abs(a - b));
  }
}

std::mt19937_64 rng(42);  // fixed seed

double uniform(double lo, double hi) {
  return std::uniform_real_distribution<double>(lo, hi)(rng);
}

// ---------------------------------------------------------------------------
// T01-U-01: WGS84 <-> ENU round-trip, error < 1e-6 m up to 20 km from anchor
// ---------------------------------------------------------------------------
void testGeodeticRoundTrip() {
  std::printf("T01-U-01  WGS84 <-> ENU round-trip\n");

  // Anchors spanning equator to high latitude: the conversion is most delicate
  // near the poles, and a mid-latitude-only test would hide that.
  const std::vector<Geodetic> anchors = {
      {deg2rad(0.0), deg2rad(0.0), 0.0},        // equator
      {deg2rad(22.30), deg2rad(114.17), 12.0},  // Hong Kong (MARS-LVIG)
      {deg2rad(56.84), deg2rad(60.61), 250.0},  // Yekaterinburg
      {deg2rad(78.22), deg2rad(15.65), 40.0},   // high latitude
      {deg2rad(-33.87), deg2rad(151.21), 5.0},  // southern hemisphere
  };

  double worst = 0.0;
  for (const auto& a : anchors) {
    LocalEnu enu(a);
    for (int i = 0; i < 2000; ++i) {
      // Up to 20 km horizontally, +-2 km vertically.
      const Eigen::Vector3d p(uniform(-20000, 20000), uniform(-20000, 20000),
                              uniform(-2000, 2000));
      const Geodetic g = enu.fromEnu(p);
      const Eigen::Vector3d back = enu.toEnu(g);
      worst = std::max(worst, (back - p).norm());
    }
  }
  std::printf("  worst round-trip error: %.3e m\n", worst);
  check(worst < 1e-6, "round-trip < 1e-6 m");

  // Anchor maps to the origin exactly.
  LocalEnu enu(anchors[1]);
  const Eigen::Vector3d o = enu.toEnu(anchors[1]);
  checkNear(o.norm(), 0.0, 1e-9, "anchor maps to ENU origin");

  // Sanity of axis directions: +lat -> +North, +lon -> +East.
  Geodetic north = anchors[1];
  north.lat += deg2rad(0.01);
  check(enu.toEnu(north).y() > 0.0, "increasing latitude gives +North");
  Geodetic east = anchors[1];
  east.lon += deg2rad(0.01);
  check(enu.toEnu(east).x() > 0.0, "increasing longitude gives +East");

  // Scale sanity: 0.01 deg of latitude is ~1.11 km.
  checkNear(enu.toEnu(north).y(), 1111.0, 5.0, "latitude scale ~1.11 km / 0.01 deg");
}

// ---------------------------------------------------------------------------
// T01-U-02: SE(2) composition and inversion (property)
// ---------------------------------------------------------------------------
void testSE2() {
  std::printf("T01-U-02  SE(2) composition / inversion\n");

  double worst_inv = 0.0, worst_assoc = 0.0;
  for (int i = 0; i < 20000; ++i) {
    const SE2 a(uniform(-1e4, 1e4), uniform(-1e4, 1e4), uniform(-10.0, 10.0));
    const SE2 b(uniform(-1e4, 1e4), uniform(-1e4, 1e4), uniform(-10.0, 10.0));
    const SE2 c(uniform(-1e4, 1e4), uniform(-1e4, 1e4), uniform(-10.0, 10.0));

    // a * a^-1 == identity
    const SE2 id = a * a.inverse();
    worst_inv = std::max({worst_inv, id.translation().norm(), std::abs(id.yaw())});

    // associativity: (a*b)*c == a*(b*c)
    const SE2 l = (a * b) * c;
    const SE2 r = a * (b * c);
    worst_assoc = std::max({worst_assoc, (l.translation() - r.translation()).norm(),
                            std::abs(angleDiff(l.yaw(), r.yaw()))});

    // between(): a.between(b) applied from a recovers b
    check((a * a.between(b)).isApprox(b, 1e-6), "between() consistency");

    // matrix form agrees with the direct action on a point
    const Eigen::Vector2d p(uniform(-1e3, 1e3), uniform(-1e3, 1e3));
    const Eigen::Vector3d ph(p.x(), p.y(), 1.0);
    const Eigen::Vector3d mp = a.matrix() * ph;
    check(((a * p) - mp.head<2>()).norm() < 1e-9, "matrix() agrees with operator*");
  }
  std::printf("  worst |a*a^-1 - I|: %.3e, worst associativity: %.3e\n", worst_inv, worst_assoc);
  check(worst_inv < 1e-9, "a * a^-1 == identity within 1e-9");
  check(worst_assoc < 1e-6, "associativity within 1e-6");

  // Yaw is always stored normalised, even from a wildly out-of-range input.
  const SE2 big(0.0, 0.0, 1000.0 * kTwoPi + 0.3);
  checkNear(big.yaw(), 0.3, 1e-9, "yaw normalised at construction");
}

// ---------------------------------------------------------------------------
// T01-U-03: angle normalisation (property)
// ---------------------------------------------------------------------------
void testAngles() {
  std::printf("T01-U-03  angle normalisation\n");

  for (int i = 0; i < 50000; ++i) {
    const double a = uniform(-1e5, 1e5);
    const double n = normalizeAngle(a);
    check(n >= -kPi - 1e-12 && n <= kPi + 1e-12, "normalised into [-pi, pi]");
    // Equivalent modulo 2*pi: compare on the unit circle, which avoids
    // catastrophic cancellation for large inputs.
    const double d = std::hypot(std::cos(a) - std::cos(n), std::sin(a) - std::sin(n));
    check(d < 1e-9, "normalisation preserves the angle mod 2pi");
  }

  // Idempotence.
  for (int i = 0; i < 10000; ++i) {
    const double a = uniform(-1e4, 1e4);
    checkNear(normalizeAngle(normalizeAngle(a)), normalizeAngle(a), 0.0, "idempotent");
  }

  // Boundary: -pi and +pi share a single representation.
  checkNear(normalizeAngle(-kPi), kPi, 1e-12, "-pi maps to +pi");
  checkNear(normalizeAngle(kPi), kPi, 1e-12, "+pi stays +pi");

  // angleDiff is antisymmetric away from the wrap point.
  for (int i = 0; i < 10000; ++i) {
    const double a = uniform(-kPi, kPi), b = uniform(-kPi, kPi);
    const double d1 = angleDiff(a, b), d2 = angleDiff(b, a);
    if (std::abs(std::abs(d1) - kPi) > 1e-6) {
      checkNear(d1, -d2, 1e-9, "angleDiff antisymmetric");
    }
  }

  // Circular mean is correct across the wrap point, where an arithmetic mean
  // would give 180 deg instead of 0.
  const std::vector<double> wrapped = {deg2rad(179.0), deg2rad(-179.0)};
  checkNear(std::abs(circularMean(wrapped.begin(), wrapped.end())), kPi, 1e-6,
            "circular mean across wrap");
  const std::vector<double> near_zero = {deg2rad(-1.0), deg2rad(1.0), deg2rad(0.0)};
  checkNear(circularMean(near_zero.begin(), near_zero.end()), 0.0, 1e-9,
            "circular mean near zero");
}

// ---------------------------------------------------------------------------
// T01-U-04: raster pixel <-> ENU round-trip and windowing
// ---------------------------------------------------------------------------
void testRaster() {
  std::printf("T01-U-04  raster pixel <-> ENU\n");

  // A 10 x 2 km corridor at 0.5 m/px -- the real basemap shape.
  const GeoRaster map(500000.0, 6300000.0, 0.5, 20000, 4000);

  double worst = 0.0;
  for (int i = 0; i < 20000; ++i) {
    const double col = uniform(0.0, 20000.0), row = uniform(0.0, 4000.0);
    const Eigen::Vector2d enu = map.pixelToEnu(col, row);
    const Eigen::Vector2d back = map.enuToPixel(enu);
    worst = std::max(worst, (back - Eigen::Vector2d(col, row)).norm());
  }
  std::printf("  worst pixel round-trip: %.3e px\n", worst);
  check(worst < 1e-9, "pixel round-trip < 1e-9 px");

  // Corner conventions: (0,0) is the top-left, north decreases with row.
  const Eigen::Vector2d tl = map.pixelToEnu(0.0, 0.0);
  checkNear(tl.x(), 500000.0, 1e-9, "origin east at pixel (0,0)");
  checkNear(tl.y(), 6300000.0, 1e-9, "origin north at pixel (0,0)");
  check(map.pixelToEnu(0.0, 100.0).y() < tl.y(), "row increases southward");
  check(map.pixelToEnu(100.0, 0.0).x() > tl.x(), "col increases eastward");

  // Pixel centres sit half a GSD inside the corner.
  const Eigen::Vector2d c00 = map.pixelCenterToEnu(0, 0);
  checkNear(c00.x(), 500000.0 + 0.25, 1e-9, "pixel centre east");
  checkNear(c00.y(), 6300000.0 - 0.25, 1e-9, "pixel centre north");

  // Bounds.
  const Eigen::Vector4d b = map.boundsEnu();
  checkNear(b[2] - b[0], 10000.0, 1e-9, "corridor is 10 km wide");
  checkNear(b[3] - b[1], 2000.0, 1e-9, "corridor is 2 km tall");

  // Sub-window keeps geo-referencing: a point's ENU must be identical whether
  // read through the parent or the crop. This is where offsets usually creep in.
  const GeoRaster sub = map.subWindow(1234, 567, 600, 600);
  const Eigen::Vector2d via_parent = map.pixelToEnu(1234 + 250.0, 567 + 300.0);
  const Eigen::Vector2d via_sub = sub.pixelToEnu(250.0, 300.0);
  check((via_parent - via_sub).norm() < 1e-9, "sub-window preserves geo-referencing");

  // Prior-window request (ADR-005 shape), clamped at the raster edge.
  const Eigen::Vector2d center = map.pixelCenterToEnu(10000, 2000);
  const GeoRaster win = map.windowAround(center, 300.0);
  checkNear(win.width() * win.gsd(), 600.0, 1.0, "300 m radius window is ~600 m wide");
  check(win.containsEnu(center), "window contains its centre");

  const GeoRaster edge = map.windowAround(map.pixelCenterToEnu(5, 5), 300.0);
  check(edge.width() > 0 && edge.height() > 0, "window at the edge is clamped, not empty");
  check(edge.width() <= map.width(), "clamped window stays inside the map");

  // Coarser pyramid level covers the same extent.
  const GeoRaster coarse = map.atGsd(4.0);
  const Eigen::Vector4d cb = coarse.boundsEnu();
  checkNear(cb[2] - cb[0], 10000.0, 1e-6, "pyramid level keeps geographic extent");

  // Outside the mission package is a NORMAL situation, handled, not crashing.
  check(!map.containsEnu({400000.0, 6300000.0}), "point outside is reported outside");
}

// ---------------------------------------------------------------------------
// T01-U-05: covariance validation, systematic term, NEES
// ---------------------------------------------------------------------------
void testCovariance() {
  std::printf("T01-U-05  covariance validation\n");

  Cov3 good = Cov3::Zero();
  good.diagonal() << 25.0, 25.0, deg2rad(1.0) * deg2rad(1.0);
  check(isValidCovariance(good), "diagonal covariance is valid");
  check(isSymmetric(good), "diagonal covariance is symmetric");

  // Asymmetric input is detected, and symmetrize() repairs it.
  Cov3 asym = good;
  asym(0, 1) = 1.0;
  asym(1, 0) = 2.0;
  check(!isSymmetric(asym), "asymmetry detected");
  check(isSymmetric(symmetrize(asym)), "symmetrize() repairs asymmetry");

  // Non-positive-definite is rejected, and symmetrize() does NOT fix it.
  Cov3 npd = Cov3::Zero();
  npd.diagonal() << 1.0, -1.0, 1.0;
  check(!isPositiveDefinite(npd), "negative eigenvalue rejected");
  check(!isValidCovariance(symmetrize(npd)), "symmetrize() does not rescue non-PD");

  // NaN / inf rejected.
  Cov3 nan_cov = good;
  nan_cov(0, 0) = std::nan("");
  check(!isValidCovariance(nan_cov), "NaN rejected");

  // --- The systematic term must NOT shrink with inlier count ---
  // This is the single most common modelling mistake (see covariance.hpp).
  const double sigma_bias = 3.0;  // basemap georeferencing bias, metres
  double eph_prev = 1e9;
  for (int n : {30, 120, 480, 1920, 100000}) {
    Cov3 random_part = Cov3::Zero();
    // Random part scales as 1/n.
    random_part.diagonal() << 100.0 / n, 100.0 / n, (deg2rad(2.0) * deg2rad(2.0)) / n;
    const Cov3 total = addSystematic(random_part, sigma_bias);
    const double eph = ephFromCovariance(total);
    check(eph < eph_prev, "eph decreases with inlier count");
    eph_prev = eph;
  }
  // In the limit, eph tends to the bias floor, not to zero.
  const double floor_eph = std::sqrt(2.0) * sigma_bias;
  std::printf("  eph at n=100000: %.4f m, bias floor: %.4f m\n", eph_prev, floor_eph);
  check(std::abs(eph_prev - floor_eph) < 0.05, "eph converges to the systematic floor");
  check(eph_prev > 0.9 * floor_eph, "systematic term never washed out by inliers");

  // Floor on the diagonal.
  Cov3 tiny = Cov3::Zero();
  tiny.diagonal() << 1e-8, 1e-8, 1e-10;
  const Cov3 floored = applyFloor(tiny, 0.5, deg2rad(0.1));
  checkNear(std::sqrt(floored(0, 0)), 0.5, 1e-9, "position floor applied");
  checkNear(std::sqrt(floored(2, 2)), deg2rad(0.1), 1e-12, "yaw floor applied");

  // Prior window radius (ADR-005), with clamping.
  Cov3 wide = Cov3::Zero();
  wide.diagonal() << 400.0, 400.0, deg2rad(3.0) * deg2rad(3.0);  // sigma = 20 m each
  const double r = priorWindowRadius(wide, 20.0, 30.0, 1000.0);
  checkNear(r, 3.0 * std::sqrt(800.0) + 20.0, 1e-9, "R = 3*eph + margin");
  checkNear(priorWindowRadius(good, 0.0, 30.0, 1000.0), 30.0, 1e-9, "R clamped to minimum");
  Cov3 huge = Cov3::Identity() * 1e8;
  checkNear(priorWindowRadius(huge, 0.0, 30.0, 1000.0), 1000.0, 1e-9, "R clamped to maximum");

  // --- NEES: the honesty check for the covariance model ---
  // With a correctly calibrated covariance, mean NEES over many samples equals
  // the number of degrees of freedom (3). An understated covariance inflates it.
  {
    Cov3 P = Cov3::Zero();
    P.diagonal() << 9.0, 4.0, deg2rad(1.0) * deg2rad(1.0);
    const Eigen::Vector3d sd(std::sqrt(P(0, 0)), std::sqrt(P(1, 1)), std::sqrt(P(2, 2)));
    std::normal_distribution<double> gauss(0.0, 1.0);

    double sum_correct = 0.0, sum_understated = 0.0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
      const Eigen::Vector3d e(sd.x() * gauss(rng), sd.y() * gauss(rng), sd.z() * gauss(rng));
      sum_correct += nees(e, P);
      sum_understated += nees(e, (P * 0.25).eval());  // sigma halved
    }
    const double mean_correct = sum_correct / N;
    const double mean_understated = sum_understated / N;
    std::printf("  mean NEES: correct %.3f (expect 3.0), understated %.3f (expect 12.0)\n",
                mean_correct, mean_understated);
    check(std::abs(mean_correct - 3.0) < 0.05, "NEES == DoF for an honest covariance");
    check(mean_understated > 11.0, "NEES inflates for an understated covariance");
  }

  // Chi-squared gate: 3 DoF, 99% threshold is 11.34.
  {
    Cov3 P = Cov3::Identity();
    check(mahalanobisSq({1.0, 1.0, 1.0}, P) < 11.34, "small residual passes the chi2 gate");
    check(mahalanobisSq({10.0, 10.0, 10.0}, P) > 11.34, "large residual fails the chi2 gate");
  }
}

}  // namespace

int main() {
  std::printf("geoloc_common property tests (T01-U-01 .. T01-U-05)\n");
  std::printf("====================================================\n");
  testGeodeticRoundTrip();
  testSE2();
  testAngles();
  testRaster();
  testCovariance();
  std::printf("====================================================\n");
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) std::printf("ALL PASS\n");
  return g_failures == 0 ? 0 : 1;
}
