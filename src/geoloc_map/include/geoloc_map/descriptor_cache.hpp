// Copyright 2026 geoloc team.
// Map-side descriptor cache (ADR-005).
//
// Map-window descriptors are computed ON BOARD and cached -- never precomputed
// offline, because offline precomputation ties the map package to a model
// version and inflates it. The cache is invalidated when:
//
//   1. the window centre shifts by more than `invalidate_on_shift_fraction` of
//      the window size (30%),
//   2. the requested GSD changes (different pyramid level -> different image),
//   3. the matcher model version changes -- the one everybody forgets, which
//      produces a silent descriptor mismatch after a model update.
//
// The descriptor computation itself is owned by the matcher's model (T16-T20).
// This node provides it through the injectable DescriptorEngine interface; the
// default engine is a deterministic placeholder so the cache and its
// invalidation are real and testable before XFeat lands.

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace geoloc {

/// Descriptor set in the MapWindow.srv layout.
struct DescriptorSet {
  uint32_t n_keypoints{0};
  std::vector<float> keypoints_xy;  // 2*N, window pixel coordinates
  std::vector<int8_t> descriptors;  // N*64, int8
};

/// Abstract descriptor extractor. Implementations are deterministic: the same
/// window must produce the same descriptors, or the cache is meaningless.
class DescriptorEngine {
 public:
  virtual ~DescriptorEngine() = default;
  virtual DescriptorSet compute(const std::vector<uint8_t>& gray, int width, int height,
                                const std::vector<uint8_t>& validity) = 0;
};

/// Deterministic placeholder: keypoints on a regular grid (skipping invalid
/// pixels), each descriptor is a 64-bit binarised local 8x8 patch. This makes
/// the cache logic fully testable; the real XFeat engine replaces it in T16-T20.
class GridDescriptorEngine : public DescriptorEngine {
 public:
  explicit GridDescriptorEngine(int stride = 16) : stride_(stride < 1 ? 1 : stride) {}

  DescriptorSet compute(const std::vector<uint8_t>& gray, int width, int height,
                        const std::vector<uint8_t>& validity) override {
    DescriptorSet out;
    if (width <= 0 || height <= 0 || gray.empty()) return out;

    const int patch = 8;
    for (int y = stride_ / 2; y < height; y += stride_) {
      for (int x = stride_ / 2; x < width; x += stride_) {
        if (!validity.empty() && validity[y * width + x] == 0) continue;
        // Binarise the local patch into 64 bits; average the patch as threshold
        // so the descriptor is invariant to uniform gain.
        int sum = 0;
        int cnt = 0;
        std::vector<int8_t> bits(patch * patch);
        for (int dy = 0; dy < patch; ++dy) {
          for (int dx = 0; dx < patch; ++dx) {
            int px = x + dx - patch / 2;
            int py = y + dy - patch / 2;
            if (px < 0 || px >= width || py < 0 || py >= height) {
              bits[dy * patch + dx] = 0;
              continue;
            }
            const int v = gray[py * width + px];
            bits[dy * patch + dx] = static_cast<int8_t>(v);
            sum += v;
            ++cnt;
          }
        }
        const int mean = cnt > 0 ? sum / cnt : 0;
        int8_t desc[64];
        for (int i = 0; i < 64; ++i) desc[i] = (bits[i] >= mean) ? 1 : -1;

        out.keypoints_xy.push_back(static_cast<float>(x));
        out.keypoints_xy.push_back(static_cast<float>(y));
        out.descriptors.insert(out.descriptors.end(), desc, desc + 64);
      }
    }
    out.n_keypoints = static_cast<uint32_t>(out.keypoints_xy.size() / 2);
    return out;
  }

 private:
  int stride_;
};

/// Tracks the cached descriptor set and answers validity queries.
class DescriptorCache {
 public:
  DescriptorCache(double shift_fraction, bool invalidate_on_model_change)
      : shift_fraction_(shift_fraction), invalidate_on_model_change_(invalidate_on_model_change) {}

  /// True when the cached set can be reused for this request.
  bool valid(const Eigen::Vector2d& center, double radius_m, double served_gsd,
             const std::string& model_version) const noexcept {
    if (!has_value_) return false;
    if (invalidate_on_model_change_ && model_version != model_version_) return false;
    if (served_gsd != gsd_) return false;
    const double window = 2.0 * radius_m;
    const double threshold = shift_fraction_ * window;
    if ((center - center_).norm() > threshold) return false;
    return true;
  }

  void store(const Eigen::Vector2d& center, double radius_m, double served_gsd,
             const std::string& model_version, DescriptorSet descriptors) {
    center_ = center;
    radius_ = radius_m;
    gsd_ = served_gsd;
    model_version_ = model_version;
    descriptors_ = std::move(descriptors);
    has_value_ = true;
  }

  const DescriptorSet& descriptors() const noexcept { return descriptors_; }
  bool hasValue() const noexcept { return has_value_; }

  /// Number of descriptor computations performed since construction -- the
  /// metric the T07-U-03 cache hit test asserts on.
  uint64_t recomputeCount() const noexcept { return recompute_count_; }
  void incrementRecomputeCount() noexcept { ++recompute_count_; }

 private:
  double shift_fraction_;
  bool invalidate_on_model_change_;
  bool has_value_{false};
  Eigen::Vector2d center_{Eigen::Vector2d::Zero()};
  double radius_{0.0};
  double gsd_{0.0};
  std::string model_version_;
  DescriptorSet descriptors_;
  uint64_t recompute_count_{0};
};

}  // namespace geoloc
