#!/usr/bin/env python3
"""
Production In-L3 Zstandard Dictionary Compression Wrapper via ctypes (libzstd.so.1)
Compresses and decompresses short JSONL agent transcripts and ephemeral /dev/shm
command outputs using a 112 KiB L2/L3-resident Zstandard dictionary.
"""

from __future__ import annotations

import ctypes

STREAM_SEED = 0x732641


class L3ZstdDictCompressor:
    """Zero-dependency ctypes binding to libzstd.so.1 dictionary compression."""

    def __init__(self, dict_bytes: bytes, level: int = 3):
        self.dict_bytes = dict_bytes
        self.level = level
        self._lib = ctypes.CDLL("libzstd.so.1")

        self._lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self._lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self._lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self._lib.ZSTD_isError.restype = ctypes.c_uint

        self._lib.ZSTD_createCCtx.argtypes = []
        self._lib.ZSTD_createCCtx.restype = ctypes.c_void_p
        self._lib.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
        self._lib.ZSTD_freeCCtx.restype = ctypes.c_size_t
        self._lib.ZSTD_compress_usingDict.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.ZSTD_compress_usingDict.restype = ctypes.c_size_t

        self._lib.ZSTD_createDCtx.argtypes = []
        self._lib.ZSTD_createDCtx.restype = ctypes.c_void_p
        self._lib.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
        self._lib.ZSTD_freeDCtx.restype = ctypes.c_size_t
        self._lib.ZSTD_decompress_usingDict.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._lib.ZSTD_decompress_usingDict.restype = ctypes.c_size_t

    def compress(self, raw_bytes: bytes) -> bytes:
        if not raw_bytes:
            return b""
        bound = self._lib.ZSTD_compressBound(len(raw_bytes))
        dst = ctypes.create_string_buffer(bound)
        cctx = self._lib.ZSTD_createCCtx()
        try:
            csize = self._lib.ZSTD_compress_usingDict(
                cctx,
                dst, bound,
                raw_bytes, len(raw_bytes),
                self.dict_bytes, len(self.dict_bytes),
                self.level,
            )
            if self._lib.ZSTD_isError(csize):
                raise RuntimeError("ZSTD_compress_usingDict failed")
            return dst.raw[:csize]
        finally:
            self._lib.ZSTD_freeCCtx(cctx)

    def decompress(self, comp_bytes: bytes, max_raw_len: int) -> bytes:
        if not comp_bytes:
            return b""
        dst = ctypes.create_string_buffer(max_raw_len)
        dctx = self._lib.ZSTD_createDCtx()
        try:
            dsize = self._lib.ZSTD_decompress_usingDict(
                dctx,
                dst, max_raw_len,
                comp_bytes, len(comp_bytes),
                self.dict_bytes, len(self.dict_bytes),
            )
            if self._lib.ZSTD_isError(dsize):
                raise RuntimeError("ZSTD_decompress_usingDict failed")
            return dst.raw[:dsize]
        finally:
            self._lib.ZSTD_freeDCtx(dctx)


if __name__ == "__main__":
    sample_dict = (
        b'{"role":"assistant","tool_calls":[{"name":"run_command","arguments":{"CommandLine":"'
        b'","WaitMsBeforeAsync":5000}}],"status":"DONE","source":"MODEL","step_index":'
    ) * 64
    codec = L3ZstdDictCompressor(sample_dict)
    line = b'{"role":"assistant","tool_calls":[{"name":"run_command","arguments":{"CommandLine":"cargo test --release","WaitMsBeforeAsync":5000}}],"status":"DONE","source":"MODEL","step_index":42}'
    compressed = codec.compress(line)
    restored = codec.decompress(compressed, len(line) * 2)
    assert restored == line
    print(f"Raw: {len(line)} B -> L3-Zstd Dict: {len(compressed)} B ({len(line)/len(compressed):.2f}x)")
