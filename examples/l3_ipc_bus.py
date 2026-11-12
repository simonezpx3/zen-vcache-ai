#!/usr/bin/env python3
"""
Production L3 Lockless Ring-Buffer IPC Bus (/dev/shm/vcache_l3_ipc)
Provides nanosecond inter-process messaging inside the 96 MB 3D V-Cache
without Linux kernel socket/pipe syscalls.

Build shared library first:
    gcc -O3 -march=znver4 -shared -fPIC examples/l3_ipc.c -o examples/libl3_ipc.so -lrt
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "libl3_ipc.so"
BUS_SEED = 0x732641

_c_ipc = None
if LIB_PATH.exists():
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.l3_ipc_publish.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.l3_ipc_publish.restype = ctypes.c_int

        lib.l3_ipc_get_write_seq.argtypes = []
        lib.l3_ipc_get_write_seq.restype = ctypes.c_uint64

        lib.l3_ipc_read_latest.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.l3_ipc_read_latest.restype = ctypes.c_int
        _c_ipc = lib
    except Exception:
        _c_ipc = None


class L3IpcBus:
    """Python ctypes interface for the lockless /dev/shm L3 ring buffer."""

    def __init__(self):
        self.available = _c_ipc is not None

    def publish_json(self, record: dict) -> int:
        """
        Publishes a JSON record into the L3 lockless ring buffer.
        Returns C-level write latency in nanoseconds (or -1 if unavailable).
        """
        if not self.available or _c_ipc is None:
            return -1
        raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        elapsed = ctypes.c_uint64(0)
        rc = _c_ipc.l3_ipc_publish(raw, len(raw), ctypes.byref(elapsed))
        if rc != 0:
            return -1
        return int(elapsed.value)

    def write_seq(self) -> int:
        """Returns the current atomic write sequence counter."""
        if not self.available or _c_ipc is None:
            return 0
        return int(_c_ipc.l3_ipc_get_write_seq())

    def read_latest(self, max_items: int = 32) -> list[dict]:
        """Reads the latest max_items verified JSON messages from the L3 ring buffer."""
        if not self.available or _c_ipc is None:
            return []
        cap = max(4096, max_items * 512)
        buf = ctypes.create_string_buffer(cap)
        count = _c_ipc.l3_ipc_read_latest(max_items, buf, cap)
        if count <= 0:
            return []
        raw_text = buf.value.decode("utf-8", errors="replace")
        out = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


if __name__ == "__main__":
    bus = L3IpcBus()
    if not bus.available:
        print("Compile examples/libl3_ipc.so first: gcc -O3 -march=znver4 -shared -fPIC examples/l3_ipc.c -o examples/libl3_ipc.so -lrt")
    else:
        ns = bus.publish_json({"event": "route_decision", "winner": "qwen35", "tier": "L3_VNNI"})
        print(f"Published message into L3 /dev/shm ring in {ns} ns (seq={bus.write_seq()})")
        print("Latest messages:", bus.read_latest(3))
