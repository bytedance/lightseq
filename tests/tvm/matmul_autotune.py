import logging
import sys

import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm import autotvm
from tvm.contrib import nvcc

# check whether the gpu has tensorcore
if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
    raise Exception("skip building this tutorial because cuda is not enabled..")

ctx = tvm.cuda()
if not nvcc.have_tensorcore(ctx.compute_version):
    raise Exception("the gpu has no tensorcore, skipping...")

M, N, L = 1024, 32, 4096
dtype = "int8"
layout = "TN"
if len(sys.argv) >= 4:
    M, N, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if len(sys.argv) >= 5:
    dtype = sys.argv[4]
if len(sys.argv) >= 6:
    layout = sys.argv[5]

# check whether current gpu arch support support current dtype's wmma codegen
cuda_compute_capability = tvm.runtime._ffi_api.GetDeviceAttr(2, 0, 4)
major, minor = nvcc.parse_compute_version(cuda_compute_capability)
if dtype == "int8":
    assert major == 7 and minor >= 2
elif dtype == "int4" or dtype == "int1":
    # int4/int1 only support layout TN
    assert major == 7 and minor == 5 and layout == "TN"


def matmul_nn(A, B, L, dtype="float16", layout="NN"):
    k = te.reduce_axis((0, L), name="k")
    if dtype == "float16":
        out_type = "float"
    elif dtype == "int8":
        out_type = "int"
    elif dtype == "int4" or dtype == "int1":
        out_type = "int"
    if layout == "NN":
        return te.compute(
            (N, M),
            lambda i, j: te.sum(
                A[i, k].astype(out_type) * B[k, j].astype(out_type), axis=k
            ),
        )
    if layout == "NT":
        return te.compute(
            (N, M),
            lambda i, j: te.sum(
                A[k, i].astype(out_type) * B[k, j].astype(out_type), axis=k
            ),
        )
    if layout == "TN":
        return te.compute(
            (N, M),
            lambda i, j: te.sum(
                A[i, k].astype(out_type) * B[j, k].astype(out_type), axis=k
            ),
        )
    if layout == "TT":
        return te.compute(
            (N, M),
            lambda i, j: te.sum(
                A[k, i].astype(out_type) * B[j, k].astype(out_type), axis=k
            ),
        )


@autotvm.template("tutorial/auto_tensorcore/test_gemm")
def test_gemm(N, L, M, dtype, layout):
    if layout == "NN":
        shape_a = (N, L)
        shape_b = (L, M)
    elif layout == "NT":
        shape_a = (L, N)
        shape_b = (L, M)
    elif layout == "TN":
        shape_a = (N, L)
        shape_b = (M, L)
    elif layout == "TT":
        shape_a = (L, N)
        shape_b = (M, L)
    else:
        print("Unsupported layout:", layout)
        sys.exit(1)
    A = te.placeholder(shape_a, name="A", dtype=dtype)
    B = te.placeholder(shape_b, name="B", dtype=dtype)
    C = matmul_nn(A, B, L, dtype, layout)

    s = te.create_schedule(C.op)
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # storage_align params
    factor = 16
    offset = 8
    if dtype == "int8":
        factor = 32
        offset = 16
    elif dtype == "int4":
        factor = 64
        offset = 32
    elif dtype == "int1":
        factor = 256
        offset = 128

    # create cache stages
    AA = s.cache_read(A, "shared", [C])
    if layout == "NN" or layout == "TN":
        s[AA].storage_align(AA.op.axis[0], factor, offset)
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    if layout == "TT" or layout == "NT":
        s[BB].storage_align(BB.op.axis[0], factor, offset)
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    # autotvm search space definition
    cfg = autotvm.get_config()

    cfg.define_knob("bx", [2, 4, 8])
    cfg.define_knob("by", [8, 16, 32, 64])
    cfg.define_knob("step_k", [1, 2, 4, 8, 16, 32])
    cfg.define_knob("v", [4, 8, 16, 32])
    by = cfg["by"].val
    bx = cfg["bx"].val
    step_k = cfg["step_k"].val
    v = cfg["v"].val

    # thread tile
    TX = 8
    TY = 1
    if dtype == "int4" or dtype == "int1":
        TX = 2
    # warp tile
    warp_tile_m = 16  # it could also be 8 or 32 on CUDA version >= 10.0
    warp_tile_k = 16  # it must be 16 for fp16/int8 data type
    if dtype == "int4":
        warp_tile_m = 8
        warp_tile_k = 32
    elif dtype == "int1":
        warp_tile_m = 8
        warp_tile_k = 128
    # block tile
    tile_x = bx * TX
    tile_y = by * TY

    yo, ty = s[C].split(y, tile_y)
    ty, yi = s[C].split(ty, TY)

    # schedule for C stage
    xo, xi = s[C].split(x, tile_x)
    WX = min(warp_tile_m, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    s[C].reorder(yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    # schedule for CL stage
    ko, ki = s[CL].split(k, step_k * warp_tile_k)
    kl, ki = s[CL].split(ki, warp_tile_k)
    s[CL].compute_at(s[C], tx)
    yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, yo, xo)

    # schedule for AA stage
    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[1], factor=bx * v)
    tz, tx = s[AA].split(xi, factor=(WX // TX) * v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[0], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tz, te.thread_axis("threadIdx.z"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    # vectorization is very important for float16/int8 inputs
    s[AA].vectorize(vec)

    # schedule for BB stage
    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[1], factor=bx * v)
    tz, tx = s[BB].split(xi, factor=(WX // TX) * v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[0], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tz, te.thread_axis("threadIdx.z"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    # set the 'tensor_core' pragma for tensorcore codegen
    s[CL].pragma(ko, "tensor_core")

    return s, [A, B, C]


def tune_and_evaluate(M, N, L, dtype, layout):
    task = autotvm.task.create(
        "tutorial/auto_tensorcore/test_gemm",
        args=(N, L, M, dtype, layout),
        target="cuda",
    )
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder="local", runner=autotvm.LocalRunner(number=5)
    )

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")],
    )

    dispatch_context = autotvm.apply_history_best("matmul.log")
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    with autotvm.apply_history_best("matmul.log"):
        with tvm.target.Target("cuda"):
            s, arg_bufs = test_gemm(N, L, M, dtype, layout)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
    dev_module = func.imported_modules[0]
    print(dev_module.get_source())

    # check correctness
    if layout == "NN":
        shape_a = (N, L)
        shape_b = (L, M)
    elif layout == "NT":
        shape_a = (L, N)
        shape_b = (L, M)
    elif layout == "TN":
        shape_a = (N, L)
        shape_b = (M, L)
    elif layout == "TT":
        shape_a = (L, N)
        shape_b = (M, L)

    a_np = None
    b_np = None
    c_np = None
    c_np_type = None
    if dtype == "float16":
        c_np_type = np.float32
        a_np = np.random.uniform(size=shape_a).astype(np.float16)
        b_np = np.random.uniform(size=shape_b).astype(np.float16)
        if layout == "NN":
            c_np = np.dot(a_np, b_np)
        elif layout == "NT":
            c_np = np.dot(a_np.T, b_np)
        elif layout == "TN":
            c_np = np.dot(a_np, b_np.T)
        elif layout == "TT":
            c_np = np.dot(a_np.T, b_np.T)
    elif dtype == "int8":
        c_np_type = np.int32
        a_np = np.random.randint(low=-128, high=127, size=shape_a).astype(np.int8)
        b_np = np.random.randint(low=-128, high=127, size=shape_b).astype(np.int8)
        if layout == "NN":
            c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32))
        elif layout == "NT":
            c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32))
        elif layout == "TN":
            c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32).T)
        elif layout == "TT":
            c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32).T)
    elif dtype == "int4":
        c_np_type = np.int32
        a_np_int = np.random.randint(low=-8, high=7, size=shape_a).astype(np.int32)
        b_np_int = np.random.randint(low=-8, high=7, size=shape_b).astype(np.int32)
        # "TN"
        c_np = np.dot(a_np_int.astype(np.int32), b_np_int.astype(np.int32).T)
        a_np = np.zeros(shape=(N, int(L / 8)), dtype=np.int32)
        b_np = np.zeros(shape=(M, int(L / 8)), dtype=np.int32)
        # a_np --> col_major
        for i in range(N):
            for j in range(int(L / 8)):
                for k in range(8):
                    a_np[i, j] = a_np[i, j] | (
                        (a_np_int[i, j * 8 + k] & 0xF) << ((7 - k) * 4)
                    )

        # b_np --> row_major
        for i in range(M):
            for j in range(int(L / 8)):
                for k in range(8):
                    b_np[i, j] = b_np[i, j] | (
                        (b_np_int[i, j * 8 + k] & 0xF) << ((7 - k) * 4)
                    )
    elif dtype == "int1":
        c_np_type = np.int32
        a_np_int = np.random.randint(low=0, high=1, size=shape_a).astype(np.int32)
        b_np_int = np.random.randint(low=0, high=1, size=shape_b).astype(np.int32)
        # "TN"
        c_np = np.dot(a_np_int.astype(np.int32), b_np_int.astype(np.int32).T)
        a_np = np.zeros(shape=(N, int(L / 32)), dtype=np.int32)
        b_np = np.zeros(shape=(M, int(L / 32)), dtype=np.int32)
        for i in range(N):
            for j in range(int(L / 32)):
                for k in range(32):
                    a_np[i, j] = a_np[i, j] | (
                        (a_np_int[i, j * 32 + k] & 0xF) << (31 - k)
                    )

        for i in range(M):
            for j in range(int(L / 32)):
                for k in range(32):
                    b_np[i, j] = b_np[i, j] | (
                        (b_np_int[i, j * 32 + k] & 0xF) << (31 - k)
                    )

    c_tvm = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np_type), device=ctx)
    a_tvm = tvm.nd.array(a_np, device=ctx)
    b_tvm = tvm.nd.array(b_np, device=ctx)
    func(a_tvm, b_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    print("Time cost of this operator: %f" % evaluator(a_tvm, b_tvm, c_tvm).mean)


if __name__ == "__main__":
    tune_and_evaluate(M, N, L, dtype, layout)
