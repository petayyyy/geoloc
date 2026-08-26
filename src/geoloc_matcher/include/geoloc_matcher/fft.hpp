// Minimal dependency-free FFT for the phase-correlation matcher (T19).
//
// A hand-rolled radix-2 Cooley-Tukey FFT, power-of-two sizes only. Deliberately
// dependency-free (no FFTW, no kissfft) so the phase-correlation channel and its
// level-0 tests build in a bare cross-build container, exactly like the
// geoloc_common property tests.
//
// Conventions match numpy.fft so the C++ port is line-checkable against the
// Python reference: forward = exp(-2*pi*i*k*n/N), inverse = exp(+...)/N.

#pragma once

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace geoloc {

/// In-place 1D radix-2 FFT. `n` MUST be a power of two. `inverse` normalises by
/// 1/n (numpy convention), which is what makes fft2/ifft2 round-trip identity.
inline void fft1d(std::complex<double>* a, int n, bool inverse) {
  // Bit-reversal permutation.
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  // Iterative butterflies.
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = (inverse ? 2.0 : -2.0) * std::acos(-1.0) / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int k = 0; k < len / 2; ++k) {
        const std::complex<double> u = a[i + k];
        const std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    const double inv = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i) a[i] *= inv;
  }
}

/// 2D FFT over a row-major buffer of `w * h` elements; both dimensions must be
/// powers of two. Column transforms go through a scratch buffer (columns are
/// strided, so an in-place column FFT is not contiguous).
inline void fft2(std::complex<double>* data, int w, int h, bool inverse) {
  std::vector<std::complex<double>> col(h);
  for (int y = 0; y < h; ++y) fft1d(data + y * w, w, inverse);
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < h; ++y) col[y] = data[y * w + x];
    fft1d(col.data(), h, inverse);
    for (int y = 0; y < h; ++y) data[y * w + x] = col[y];
  }
}

/// Smallest power of two >= n.
inline int nextPow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

}  // namespace geoloc
