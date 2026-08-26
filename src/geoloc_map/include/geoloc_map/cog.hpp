// Copyright 2026 geoloc team.
// COG (Cloud Optimized GeoTIFF) reader via mmap + libtiff.
//
// The .geopack mosaic is a tiled COG: 256x256 tiles, JPEG (3-band YCbCr) for
// the orthophoto and DEFLATE (1-band uint8) for the validity mask / semantics,
// with an overview pyramid (factors 2, 4, 8) stored as chained IFDs. The whole
// file is mmap'ed so a cold tile is paged in lazily by the OS -- the tens of
// milliseconds the prefetch path exists to hide.
//
// libtiff is used only for the container decoding (JPEG/DEFLATE); tile layout,
// overview walk and georeferencing are read here so the node controls exactly
// what gets decoded and when. The GeoTIFF georeferencing tags (33550
// ModelPixelScale, 33922 ModelTiepoint) are read by tag number because this
// libtiff build has no libgeotiff; the corresponding "Unknown field" warnings
// are filtered out of the log.

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tiffio.h>

namespace geoloc {

class CogReader {
 public:
  struct Level {
    int dir_index{0};  // TIFF directory index (0 == base)
    uint32_t width{0};
    uint32_t height{0};
    uint32_t tile_w{0};
    uint32_t tile_h{0};
    uint16_t samples{0};
    uint16_t photometric{0};
    double gsd{0.0};           // m/px at this level
    double origin_east{0.0};   // CRS easting of the top-left corner
    double origin_north{0.0};  // CRS northing of the top-left corner
  };

  CogReader() = default;
  ~CogReader();
  CogReader(CogReader&& o) noexcept;
  CogReader& operator=(CogReader&& o) noexcept;
  CogReader(const CogReader&) = delete;
  CogReader& operator=(const CogReader&) = delete;

  /// Open and index a COG. Throws std::runtime_error on failure.
  static CogReader open(const std::string& path);

  const std::vector<Level>& levels() const noexcept { return levels_; }
  bool valid() const noexcept { return tif_ != nullptr; }

  /// Read a pixel-aligned rectangle at `level_idx` as 8-bit grayscale (3-band
  /// sources are converted with the BT.601 luminance weights).
  bool readGray(int level_idx, int x0, int y0, int w, int h, uint8_t* out);
  /// Read a pixel-aligned rectangle at `level_idx` as single-band uint8.
  bool readUint8(int level_idx, int x0, int y0, int w, int h, uint8_t* out);
  /// Read a pixel-aligned rectangle at `level_idx` as single-band float32 (DEM).
  bool readFloat32(int level_idx, int x0, int y0, int w, int h, float* out);

  /// Decode (and discard) every tile overlapping the rectangle, warming the OS
  /// page cache and the JPEG/DEFLATE decode cache. Returns the tile count.
  int prefetchTiles(int level_idx, int x0, int y0, int w, int h);

 private:
  struct MappedFile {
    int fd{-1};
    uint8_t* data{nullptr};
    uint64_t size{0};
    uint64_t offset{0};
    ~MappedFile() { release(); }
    void release() {
      if (data) {
        munmap(data, size);
        data = nullptr;
      }
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
    }
  };

  // libtiff I/O callbacks backed by the mmap'ed file.
  static tmsize_t readProc(thandle_t h, void* buf, tmsize_t size);
  static tmsize_t writeProc(thandle_t, void*, tmsize_t);
  static toff_t seekProc(thandle_t h, toff_t off, int whence);
  static int closeProc(thandle_t);
  static toff_t sizeProc(thandle_t h);
  static int mapProc(thandle_t h, void** base, toff_t* size);
  static void unmapProc(thandle_t, void*, toff_t);

  /// Drop libtiff's "Unknown field with tag ..." noise while keeping real
  /// warnings. Installed once, process-wide.
  static void installWarningHandler();

  bool readTiles(int level_idx, int x0, int y0, int w, int h, int mode, void* out);
  // mode: 0 = gray from RGB, 1 = uint8 direct, 2 = float32 direct

  // Heap-allocated so its address stays stable across CogReader moves: the TIFF
  // handle keeps a raw pointer to it as its client data.
  std::unique_ptr<MappedFile> file_;
  TIFF* tif_{nullptr};
  std::vector<Level> levels_;
  std::vector<uint8_t> tile_buf_;
};

}  // namespace geoloc
