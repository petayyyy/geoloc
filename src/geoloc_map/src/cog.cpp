// Copyright 2026 geoloc team.
// COG reader implementation (see cog.hpp).

#include "geoloc_map/cog.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

namespace geoloc {

namespace {
constexpr int kModeGrayRgb = 0;
constexpr int kModeUint8 = 1;
constexpr int kModeFloat32 = 2;

uint16_t kGeoTiffPixelScale = 33550;  // ModelPixelScaleTag
uint16_t kGeoTiffTiepoints = 33922;   // ModelTiepointTag
}  // namespace

CogReader::~CogReader() {
  if (tif_) TIFFClose(tif_);
  tif_ = nullptr;
}

CogReader::CogReader(CogReader&& o) noexcept
    : file_(std::move(o.file_)),
      tif_(o.tif_),
      levels_(std::move(o.levels_)),
      tile_buf_(std::move(o.tile_buf_)) {
  o.tif_ = nullptr;
}

CogReader& CogReader::operator=(CogReader&& o) noexcept {
  if (this != &o) {
    if (tif_) TIFFClose(tif_);
    file_ = std::move(o.file_);
    tif_ = o.tif_;
    levels_ = std::move(o.levels_);
    tile_buf_ = std::move(o.tile_buf_);
    o.tif_ = nullptr;
  }
  return *this;
}

void CogReader::installWarningHandler() {
  static bool installed = false;
  if (installed) return;
  installed = true;
  TIFFSetWarningHandler([](const char*, const char* fmt, va_list ap) {
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    // The GeoTIFF georeferencing tags (33550/33922/34735/34737) are read
    // explicitly by tag number; libtiff flags them as "unknown" because
    // libgeotiff is not linked. Filter that noise, keep real warnings.
    const std::string s(buf);
    if (s.find("Unknown field with tag") != std::string::npos) return;
    std::fprintf(stderr, "TIFF warning: %s\n", buf);
  });
}

CogReader CogReader::open(const std::string& path) {
  installWarningHandler();

  CogReader r;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open COG: " + path);

  struct stat st {};
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat COG: " + path);
  }
  void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("cannot mmap COG: " + path);
  }

  r.file_ = std::make_unique<MappedFile>();
  r.file_->fd = fd;
  r.file_->data = static_cast<uint8_t*>(data);
  r.file_->size = static_cast<uint64_t>(st.st_size);
  r.file_->offset = 0;

  r.tif_ = TIFFClientOpen(path.c_str(), "r", r.file_.get(), readProc, writeProc, seekProc,
                          closeProc, sizeProc, nullptr, nullptr);
  if (!r.tif_) throw std::runtime_error("libtiff rejected COG: " + path);

  // Force RGB output for YCbCr JPEG sources so the decoded tiles are 3 bytes/px.
  // Only valid for JPEG-compressed sources; setting it on a DEFLATE file is a
  // harmless no-op that libtiff warns about, so set it lazily on read instead.
  // TIFFSetField(r.tif_, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);

  // Walk all directories: base + chained overviews. Field values are copied
  // out immediately after each TIFFGetField: libtiff may reallocate its custom
  // (non-GeoTIFF-aware) field storage on the next TIFFGetField call, so a
  // pointer returned here must not outlive that call.
  Level base;
  int dir = 0;
  do {
    Level lv;
    lv.dir_index = dir;
    TIFFGetField(r.tif_, TIFFTAG_IMAGEWIDTH, &lv.width);
    TIFFGetField(r.tif_, TIFFTAG_IMAGELENGTH, &lv.height);
    TIFFGetField(r.tif_, TIFFTAG_TILEWIDTH, &lv.tile_w);
    TIFFGetField(r.tif_, TIFFTAG_TILELENGTH, &lv.tile_h);
    TIFFGetField(r.tif_, TIFFTAG_SAMPLESPERPIXEL, &lv.samples);
    TIFFGetField(r.tif_, TIFFTAG_PHOTOMETRIC, &lv.photometric);

    double scale_x = 0.0, scale_y = 0.0;
    double tie_i = 0.0, tie_j = 0.0, tie_x = 0.0, tie_y = 0.0;
    bool has_scale = false;
    bool has_tie = false;
    {
      double* scale = nullptr;
      uint16_t nscale = 0;
      if (TIFFGetField(r.tif_, kGeoTiffPixelScale, &nscale, &scale) && scale && nscale >= 2) {
        scale_x = scale[0];
        scale_y = scale[1];
        has_scale = true;
      }
    }
    {
      double* tie = nullptr;
      uint16_t ntie = 0;
      if (TIFFGetField(r.tif_, kGeoTiffTiepoints, &ntie, &tie) && tie && ntie >= 6) {
        tie_i = tie[0];
        tie_j = tie[1];
        tie_x = tie[3];
        tie_y = tie[4];
        has_tie = true;
      }
    }

    if (dir == 0) {
      if (!has_scale) throw std::runtime_error("COG missing pixel scale: " + path);
      lv.gsd = std::abs(scale_x);
      // Top-left corner: X = tieX - tieI*scaleX, Y = tieY - tieJ*scaleY.
      lv.origin_east = has_tie ? tie_x - tie_i * scale_x : 0.0;
      lv.origin_north = has_tie ? tie_y - tie_j * scale_y : 0.0;
      base = lv;
    } else {
      // Overviews carry no georeferencing; the origin is preserved and the GSD
      // scales by the true dimension ratio (handles non-exact divisors).
      lv.origin_east = base.origin_east;
      lv.origin_north = base.origin_north;
      lv.gsd = base.gsd * (static_cast<double>(base.width) / static_cast<double>(lv.width));
    }
    r.levels_.push_back(lv);
    ++dir;
  } while (TIFFReadDirectory(r.tif_));

  TIFFSetDirectory(r.tif_, 0);
  return r;
}

bool CogReader::readTiles(int level_idx, int x0, int y0, int w, int h, int mode, void* out) {
  if (!tif_ || level_idx < 0 || level_idx >= static_cast<int>(levels_.size())) return false;
  const Level& L = levels_[level_idx];
  TIFFSetDirectory(tif_, L.dir_index);
  if (L.photometric == PHOTOMETRIC_YCBCR) {
    TIFFSetField(tif_, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
  }

  const size_t buf_needed = static_cast<size_t>(L.tile_w) * L.tile_h * L.samples;
  if (tile_buf_.size() < buf_needed) tile_buf_.resize(buf_needed);

  const int tx0 = x0 / static_cast<int>(L.tile_w);
  const int tx1 = (x0 + w - 1) / static_cast<int>(L.tile_w);
  const int ty0 = y0 / static_cast<int>(L.tile_h);
  const int ty1 = (y0 + h - 1) / static_cast<int>(L.tile_h);

  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      const ttile_t tile = TIFFComputeTile(tif_, tx * L.tile_w, ty * L.tile_h, 0, 0);
      if (TIFFReadEncodedTile(tif_, tile, tile_buf_.data(), static_cast<tsize_t>(-1)) < 0) {
        return false;
      }
      const int src_x0 = tx * static_cast<int>(L.tile_w);
      const int src_y0 = ty * static_cast<int>(L.tile_h);
      const int cx0 = std::max(x0, src_x0);
      const int cy0 = std::max(y0, src_y0);
      const int cx1 = std::min(x0 + w, src_x0 + static_cast<int>(L.tile_w));
      const int cy1 = std::min(y0 + h, src_y0 + static_cast<int>(L.tile_h));

      for (int y = cy0; y < cy1; ++y) {
        const int by = y - src_y0;
        for (int x = cx0; x < cx1; ++x) {
          const int bx = x - src_x0;
          const size_t si = static_cast<size_t>(by) * L.tile_w + bx;
          const size_t di = static_cast<size_t>(y - y0) * w + (x - x0);
          if (mode == kModeGrayRgb) {
            const uint8_t r = tile_buf_[si * 3];
            const uint8_t g = tile_buf_[si * 3 + 1];
            const uint8_t b = tile_buf_[si * 3 + 2];
            static_cast<uint8_t*>(out)[di] =
                static_cast<uint8_t>((299u * r + 587u * g + 114u * b + 500u) / 1000u);
          } else if (mode == kModeUint8) {
            static_cast<uint8_t*>(out)[di] = tile_buf_[si];
          } else {  // kModeFloat32
            static_cast<float*>(out)[di] = reinterpret_cast<const float*>(tile_buf_.data())[si];
          }
        }
      }
    }
  }
  return true;
}

bool CogReader::readGray(int level_idx, int x0, int y0, int w, int h, uint8_t* out) {
  return readTiles(level_idx, x0, y0, w, h, kModeGrayRgb, out);
}

bool CogReader::readUint8(int level_idx, int x0, int y0, int w, int h, uint8_t* out) {
  return readTiles(level_idx, x0, y0, w, h, kModeUint8, out);
}

bool CogReader::readFloat32(int level_idx, int x0, int y0, int w, int h, float* out) {
  return readTiles(level_idx, x0, y0, w, h, kModeFloat32, out);
}

int CogReader::prefetchTiles(int level_idx, int x0, int y0, int w, int h) {
  if (!tif_ || level_idx < 0 || level_idx >= static_cast<int>(levels_.size())) return 0;
  const Level& L = levels_[level_idx];
  TIFFSetDirectory(tif_, L.dir_index);
  if (L.photometric == PHOTOMETRIC_YCBCR) {
    TIFFSetField(tif_, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
  }

  const size_t buf_needed = static_cast<size_t>(L.tile_w) * L.tile_h * L.samples;
  if (tile_buf_.size() < buf_needed) tile_buf_.resize(buf_needed);

  const int tx0 = x0 / static_cast<int>(L.tile_w);
  const int tx1 = (x0 + w - 1) / static_cast<int>(L.tile_w);
  const int ty0 = y0 / static_cast<int>(L.tile_h);
  const int ty1 = (y0 + h - 1) / static_cast<int>(L.tile_h);

  int touched = 0;
  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      const ttile_t tile = TIFFComputeTile(tif_, tx * L.tile_w, ty * L.tile_h, 0, 0);
      if (TIFFReadEncodedTile(tif_, tile, tile_buf_.data(), static_cast<tsize_t>(-1)) < 0) {
        continue;
      }
      ++touched;
    }
  }
  return touched;
}

// ---------------------------------------------------------------------------
// libtiff I/O callbacks over the mmap'ed file.
// ---------------------------------------------------------------------------

tmsize_t CogReader::readProc(thandle_t h, void* buf, tmsize_t size) {
  MappedFile* mf = reinterpret_cast<MappedFile*>(h);
  if (size < 0) return -1;
  uint64_t avail = mf->size > mf->offset ? mf->size - mf->offset : 0;
  uint64_t n = static_cast<uint64_t>(size) < avail ? static_cast<uint64_t>(size) : avail;
  std::memcpy(buf, mf->data + mf->offset, n);
  mf->offset += n;
  return static_cast<tmsize_t>(n);
}

tmsize_t CogReader::writeProc(thandle_t, void*, tmsize_t) {
  return -1;
}

toff_t CogReader::seekProc(thandle_t h, toff_t off, int whence) {
  MappedFile* mf = reinterpret_cast<MappedFile*>(h);
  switch (whence) {
    case SEEK_SET:
      mf->offset = off;
      break;
    case SEEK_CUR:
      mf->offset = static_cast<uint64_t>(static_cast<int64_t>(mf->offset) + off);
      break;
    case SEEK_END:
      mf->offset = mf->size + off;
      break;
    default:
      return -1;
  }
  return static_cast<toff_t>(mf->offset);
}

int CogReader::closeProc(thandle_t) {
  return 0;
}

toff_t CogReader::sizeProc(thandle_t h) {
  return static_cast<toff_t>(reinterpret_cast<MappedFile*>(h)->size);
}

int CogReader::mapProc(thandle_t h, void** base, toff_t* size) {
  MappedFile* mf = reinterpret_cast<MappedFile*>(h);
  *base = mf->data;
  *size = static_cast<toff_t>(mf->size);
  return 1;
}

void CogReader::unmapProc(thandle_t, void*, toff_t) {}

}  // namespace geoloc
