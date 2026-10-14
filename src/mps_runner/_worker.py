"""Captured script entry point. It has no Workflow or analysis dependencies."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
import uuid


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def initialize_cuda(expected_uuid):
    cuda = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        result = getattr(cuda, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with CUDA error {result}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError("the MPS child must see exactly one selected GPU")
    device = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), 0)
    identifier = (ctypes.c_ubyte * 16)()
    call("cuDeviceGetUuid", ctypes.byref(identifier), device)
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))
    if actual != expected_uuid:
        raise RuntimeError(f"CUDA selected {actual}, expected {expected_uuid}")
    context = ctypes.c_void_p()
    call("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
    call("cuCtxSetCurrent", context)
    return cuda, device, actual


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    directory = Path(request["directory"])
    cuda = None
    try:
        if request.get("gpu_uuid"):
            cuda = initialize_cuda(request["gpu_uuid"])
        write_json(directory / "ready.json", {"pid": os.getpid(), "gpu_uuid": cuda[2] if cuda else None})
        while not (directory / "go").exists():
            if (directory / "abort").exists():
                if cuda:
                    cuda[0].cuDevicePrimaryCtxRelease(cuda[1])
                return 126
            time.sleep(0.02)
        if request.get("probe"):
            if cuda:
                result = cuda[0].cuDevicePrimaryCtxRelease(cuda[1])
                if result:
                    raise RuntimeError(f"CUDA probe release failed: {result}")
            return 0
    except BaseException:
        traceback.print_exc()
        write_json(directory / "bootstrap-error.json", {"error": traceback.format_exc()})
        return 126

    code, error = 0, ""
    try:
        script = Path(request["script_path"])
        sys.argv = [str(script), *request["args"]]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exception:
        code = exception.code if isinstance(exception.code, int) else (0 if exception.code is None else 1)
        if code:
            error = f"script exited with code {exception.code}"
            print(error, file=sys.stderr)
    except BaseException:
        code, error = 1, traceback.format_exc()
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        write_json(directory / "done.json", {"exit_code": code, "error": error})
        # MPS clients wait for the manager to clear all contexts before exiting.
        # CPU mode uses the same handshake and ordinary process-group cleanup.
        while not (directory / "exit").exists():
            time.sleep(0.02)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
