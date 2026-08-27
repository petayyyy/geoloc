// T19 matcher as a standalone CLI (level-B runner backend).
//
// The metrics harness (T12) evaluates the phase-correlation channel with the
// SAME code the ROS node runs -- geoloc_matcher_node.cpp calls this exact
// PhaseCorrMatcher / phaseCorrToFix. Recompiling the header here rather than
// re-implementing the matcher in Python is what keeps the level-B numbers
// honest (P0 rule: feed the same code that runs on the aircraft).
//
// Protocol: raw float64 row-major grayscale images (patch / map / confidence),
// dimensions and the patch GSD as arguments, the PhaseCorrResult + PhaseCorrFix
// as one JSON object on stdout. The Python driver converts PNG/TIFF to .bin
// (see run_t19.py) and parses the JSON.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "geoloc_matcher/phase_corr.hpp"

using geoloc::GrayImage;
using geoloc::PhaseCorrConfig;
using geoloc::PhaseCorrCovarianceConfig;
using geoloc::PhaseCorrFix;
using geoloc::PhaseCorrMatcher;
using geoloc::PhaseCorrResult;
using geoloc::phaseCorrToFix;

namespace {

bool readArg(const char* name, int argc, char** argv, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

double readDoubleArg(const char* name, int argc, char** argv, double dflt) {
  std::string v;
  return readArg(name, argc, argv, v) ? std::atof(v.c_str()) : dflt;
}

bool readRaw(const std::string& path, int w, int h, std::vector<double>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  const size_t n = static_cast<size_t>(w) * h;
  out.resize(n);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(double)));
  return f.gcount() == static_cast<std::streamsize>(n * sizeof(double));
}

void printJson(const PhaseCorrResult& r, const PhaseCorrFix& fix, const PhaseCorrConfig& cfg,
               double match_ms) {
  std::printf("{\n");
  std::printf("  \"success\": %s,\n", r.success ? "true" : "false");
  std::printf("  \"shift_east_px\": %.9g,\n", r.shift_east_px);
  std::printf("  \"shift_north_px\": %.9g,\n", r.shift_north_px);
  std::printf("  \"delta_yaw_rad\": %.9g,\n", r.delta_yaw);
  std::printf("  \"scale\": %.9g,\n", r.scale);
  std::printf("  \"bad_scale\": %s,\n", r.bad_scale ? "true" : "false");
  std::printf("  \"peak_ratio\": %.9g,\n", r.peak_ratio);
  std::printf("  \"peak_fwhm_px\": %.9g,\n", r.peak_fwhm_px);
  std::printf("  \"main_peak\": %.9g,\n", r.main_peak);
  std::printf("  \"second_peak\": %.9g,\n", r.second_peak);
  std::printf("  \"valid_fraction\": %.9g,\n", r.valid_fraction);
  std::printf("  \"mean_confidence\": %.9g,\n", r.mean_confidence);
  std::printf("  \"n_valid_pixels\": %u,\n", r.n_valid_pixels);
  std::printf("  \"delta_east\": %.9g,\n", fix.delta_east);
  std::printf("  \"delta_north\": %.9g,\n", fix.delta_north);
  std::printf("  \"delta_yaw_fix_rad\": %.9g,\n", fix.delta_yaw);
  std::printf("  \"n_correspondences\": %u,\n", fix.n_correspondences);
  std::printf("  \"n_inliers\": %u,\n", fix.n_inliers);
  std::printf("  \"inlier_ratio\": %.9g,\n", fix.inlier_ratio);
  std::printf("  \"covisibility\": %.9g,\n", fix.covisibility);
  std::printf("  \"peak_ratio_fix\": %.9g,\n", fix.peak_ratio);
  std::printf("  \"residual_rms_px\": %.9g,\n", fix.residual_rms_px);
  std::printf("  \"spatial_spread\": %.9g,\n", fix.spatial_spread);
  std::printf("  \"mean_confidence_fix\": %.9g,\n", fix.mean_confidence);
  std::printf("  \"scale_check\": %.9g,\n", fix.scale_check);
  std::printf("  \"cov\": [%.9g, %.9g, %.9g, %.9g, %.9g, %.9g],\n",
              fix.covariance(0, 0), fix.covariance(0, 1), fix.covariance(0, 2),
              fix.covariance(1, 1), fix.covariance(1, 2), fix.covariance(2, 2));
  std::printf("  \"coarse_max_deg\": %.9g,\n", cfg.coarse_max_deg);
  std::printf("  \"grad_thresh_rel\": %.9g,\n", cfg.grad_thresh_rel);
  // Wall time of the match itself: no file IO, no process start. This is the
  // number the 05-metrics latency budget is about (~20 ms for this channel).
  std::printf("  \"match_ms\": %.9g\n", match_ms);
  std::printf("}\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string patch_path, map_path, conf_path;
  if (!readArg("--patch", argc, argv, patch_path) || !readArg("--map", argc, argv, map_path)) {
    std::fprintf(stderr, "usage: t19_match --patch P.bin --map M.bin [--conf C.bin] "
                          "--pw W --ph H --mw W --mh H [--gsd G]\n");
    return 2;
  }
  int pw = static_cast<int>(readDoubleArg("--pw", argc, argv, 0));
  int ph = static_cast<int>(readDoubleArg("--ph", argc, argv, 0));
  int mw = static_cast<int>(readDoubleArg("--mw", argc, argv, 0));
  int mh = static_cast<int>(readDoubleArg("--mh", argc, argv, 0));
  if (pw <= 0 || ph <= 0 || mw <= 0 || mh <= 0) {
    std::fprintf(stderr, "image dimensions must be positive\n");
    return 2;
  }
  const double gsd = readDoubleArg("--gsd", argc, argv, 1.0);

  PhaseCorrConfig cfg;
  cfg.grad_thresh_rel = readDoubleArg("--grad-thresh", argc, argv, cfg.grad_thresh_rel);
  cfg.coarse_max_deg = readDoubleArg("--coarse-max", argc, argv, cfg.coarse_max_deg);
  cfg.coarse_step_deg = readDoubleArg("--coarse-step", argc, argv, cfg.coarse_step_deg);
  cfg.nrho = static_cast<int>(readDoubleArg("--nrho", argc, argv, cfg.nrho));
  cfg.ntheta = static_cast<int>(readDoubleArg("--ntheta", argc, argv, cfg.ntheta));
  cfg.scale_check_tolerance =
      readDoubleArg("--scale-tol", argc, argv, cfg.scale_check_tolerance);

  PhaseCorrCovarianceConfig cov;
  cov.position_sigma_m = readDoubleArg("--pos-sigma", argc, argv, cov.position_sigma_m);
  cov.yaw_sigma_deg = readDoubleArg("--yaw-sigma", argc, argv, cov.yaw_sigma_deg);
  cov.min_peak_ratio = readDoubleArg("--min-peak-ratio", argc, argv, cov.min_peak_ratio);
  cov.basemap_bias_sigma_m =
      readDoubleArg("--bias-sigma", argc, argv, cov.basemap_bias_sigma_m);

  std::vector<double> pb, mb, cb;
  if (!readRaw(patch_path, pw, ph, pb) || !readRaw(map_path, mw, mh, mb)) {
    std::fprintf(stderr, "failed to read patch/map raw data\n");
    return 2;
  }
  GrayImage patch(pw, ph);
  patch.data = std::move(pb);
  GrayImage map(mw, mh);
  map.data = std::move(mb);
  GrayImage conf;
  bool have_conf = readArg("--conf", argc, argv, conf_path);
  if (have_conf && !conf_path.empty()) {
    std::vector<double> cv;
    if (!readRaw(conf_path, pw, ph, cv)) {
      std::fprintf(stderr, "failed to read confidence raw data\n");
      return 2;
    }
    conf.width = pw;
    conf.height = ph;
    conf.data = std::move(cv);
  }

  PhaseCorrMatcher matcher(cfg);
  const auto t0 = std::chrono::steady_clock::now();
  const PhaseCorrResult r = matcher.match(patch, map, have_conf ? &conf : nullptr);
  const PhaseCorrFix fix = phaseCorrToFix(r, gsd, cov);
  const double match_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  printJson(r, fix, cfg, match_ms);
  return 0;
}
