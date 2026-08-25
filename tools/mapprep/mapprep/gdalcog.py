"""COG conversion via the GDAL C API.

rasterio's writer silently falls back from COG to GTiff, and the COG driver
does not support GDALCreateCopy (gdal_translate handles it through Create()),
so the final containerization step calls GDALTranslate on the GDAL library
bundled with the rasterio wheel through ctypes. The source is a tiled,
compressed GTiff with internal overviews; the COG writer reuses that pyramid.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import rasterio

_GDAL = None


def _gdal():
    global _GDAL
    if _GDAL is not None:
        return _GDAL
    pkg_dir = Path(rasterio.__file__).parent
    lib_dirs = [pkg_dir / ".libs", Path(str(pkg_dir) + ".libs")]
    for lib_dir in lib_dirs:
        candidates = sorted(lib_dir.glob("gdal*.dll")) if lib_dir.exists() else []
        if candidates:
            gdal = ctypes.CDLL(str(candidates[0]))
            break
    else:
        raise RuntimeError("bundled GDAL library not found next to rasterio")

    gdal.GDALAllRegister()

    gdal.GDALTranslateOptionsNew.restype = ctypes.c_void_p
    gdal.GDALTranslateOptionsNew.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_void_p,
    ]
    gdal.GDALTranslateOptionsFree.argtypes = [ctypes.c_void_p]
    gdal.GDALTranslate.restype = ctypes.c_void_p
    gdal.GDALTranslate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    gdal.GDALOpenEx.restype = ctypes.c_void_p
    gdal.GDALOpenEx.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    gdal.GDALClose.argtypes = [ctypes.c_void_p]
    gdal.CPLGetLastErrorMsg.restype = ctypes.c_char_p
    _GDAL = gdal
    return gdal


_GDAL_OF_READONLY = 0x00


def _string_array(args: list[str]):
    buffers = [arg.encode("utf-8") for arg in args] + [None]
    return (ctypes.c_char_p * len(buffers))(*buffers)


def to_cog(src: Path, dst: Path, compress: str = "JPEG") -> None:
    with rasterio.Env():
        _to_cog_impl(src, dst, compress)


def _to_cog_impl(src: Path, dst: Path, compress: str) -> None:
    gdal = _gdal()
    src_ds = gdal.GDALOpenEx(str(src).encode("utf-8"), _GDAL_OF_READONLY, None, None, None)
    if not src_ds:
        raise RuntimeError(f"GDALOpenEx failed for {src}: {_last_error(gdal)}")
    try:
        args = [
            "-of",
            "COG",
            "-co",
            "BLOCKSIZE=256",
            "-co",
            f"COMPRESS={compress}",
            "-co",
            "OVERVIEW_RESAMPLING=AVERAGE",
            "-co",
            "OVERVIEWS=IGNORE_EXISTING",
            "-co",
            "OVERVIEW_COUNT=3",
            "-co",
            "NUM_THREADS=1",
        ]
        if compress == "JPEG":
            args += ["-co", "QUALITY=85"]
        options = gdal.GDALTranslateOptionsNew(_string_array(args), None)
        if not options:
            raise RuntimeError(f"GDALTranslateOptionsNew failed: {_last_error(gdal)}")
        try:
            usage_error = ctypes.c_int()
            dst_ds = gdal.GDALTranslate(
                str(dst).encode("utf-8"), src_ds, options, ctypes.byref(usage_error)
            )
            if not dst_ds or usage_error.value:
                raise RuntimeError(
                    f"GDALTranslate failed for {dst} (usage_error={usage_error.value}): "
                    f"{_last_error(gdal)}"
                )
            gdal.GDALClose(dst_ds)
        finally:
            gdal.GDALTranslateOptionsFree(options)
    finally:
        gdal.GDALClose(src_ds)


def _last_error(gdal) -> str:
    message = gdal.CPLGetLastErrorMsg()
    return message.decode("utf-8", errors="replace") if message else "unknown error"
