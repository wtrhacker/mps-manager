"""Small real CUDA kernel for bounded MPS lifecycle acceptance, not a benchmark."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=1)
parser.add_argument("--value", type=float, default=64)
parser.add_argument("--fail", action="store_true")
parser.add_argument("--label", default="cuda")
args = parser.parse_args()
cuda = C.CDLL("libcuda.so.1")


def call(name, *values):
    result = getattr(cuda, name)(*values)
    if result:
        raise RuntimeError(f"{name}: CUDA error {result}")


device, multiprocessors = C.c_int(), C.c_int()
call("cuCtxGetDevice", C.byref(device))
call("cuDeviceGetAttribute", C.byref(multiprocessors), C.c_int(16), device)


ptx = b"""
.version 7.0
.target sm_70
.address_size 64
.visible .entry fill(.param .u64 output, .param .u32 length, .param .f32 value) {
    .reg .pred p;
    .reg .b32 r<5>;
    .reg .b64 d<4>;
    .reg .f32 f;
    ld.param.u64 d0, [output];
    ld.param.u32 r0, [length];
    ld.param.f32 f, [value];
    mov.u32 r1, %ctaid.x;
    mov.u32 r2, %ntid.x;
    mov.u32 r3, %tid.x;
    mad.lo.u32 r4, r1, r2, r3;
    setp.ge.u32 p, r4, r0;
    @p bra end;
    mul.wide.u32 d1, r4, 4;
    add.u64 d2, d0, d1;
    st.global.f32 [d2], f;
end:
    ret;
}
"""
module, function = C.c_void_p(), C.c_void_p()
call("cuModuleLoadData", C.byref(module), C.c_char_p(ptx))
call("cuModuleGetFunction", C.byref(function), module, b"fill")
pointer, count, value = C.c_uint64(), C.c_uint32(65536), C.c_float(args.value)
call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(count.value * 4))
parameters = (C.c_void_p * 3)(C.cast(C.byref(pointer), C.c_void_p), C.cast(C.byref(count), C.c_void_p), C.cast(C.byref(value), C.c_void_p))
sample = C.c_float()
directory = Path(os.environ["MPS_RUN_DIR"])
print(json.dumps({"pid": os.getpid(), "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "label": args.label}), flush=True)
started, ticks = time.monotonic(), 0
try:
    while time.monotonic() - started < args.seconds:
        call("cuLaunchKernel", function, 256, 1, 1, 256, 1, 1, 0, None, parameters, None)
        call("cuCtxSynchronize")
        call("cuMemcpyDtoH_v2", C.byref(sample), pointer, C.c_size_t(4))
        if sample.value != args.value:
            raise AssertionError((sample.value, args.value))
        ticks += 1
        (directory / "ticks.txt").write_text(str(ticks))
        time.sleep(0.03)
    if args.fail:
        raise ValueError("deliberate CUDA-script Python failure")
    (directory / "result.json").write_text(json.dumps({"value": sample.value, "ticks": ticks, "pid": os.getpid(),
        "multiprocessors": multiprocessors.value,
        "active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}))
    print("completed CUDA work", flush=True)
finally:
    call("cuMemFree_v2", pointer)
    call("cuModuleUnload", module)
