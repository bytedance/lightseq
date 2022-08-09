import argparse
import os
import functools
from dataclasses import dataclass
from copy import deepcopy

FLOAT_MAX = float(1e9)
STRIDE = 32
BORDER = 512
MAX_BSZ = 10000


@dataclass
class GemmAlgoInfo:
    sm: int = 80
    shape: tuple = (1, 1024, 1024)
    data_order: str = "CUBLASLT_ORDER_COL"
    algo_id: int = 0
    tile: int = 0
    splitk: int = 0
    reduc: int = 0
    swizzle: int = 0
    custom: int = 0
    stages: int = 0
    workspace: int = 0
    fp16_time: float = FLOAT_MAX
    int8_time: float = FLOAT_MAX
    speedup: float = 0


def base_nk(h, i):
    return [(3 * h, h), (h, h), (i, h), (h, i)]


def rm(file):
    if os.path.exists(file):
        os.remove(file)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def get_sm():
    tmp_output_file = "tmp_output.log"
    os.system("nvcc -o get_gpu_info get_gpu_info.cpp")
    os.system("./get_gpu_info > {}".format(tmp_output_file))

    with open(tmp_output_file, "r") as fin:
        sm = int(fin.readline())
        assert sm > 0
    rm(tmp_output_file)

    return sm


def extract(log):
    best_gemm_algos = []
    col32_algo_info = GemmAlgoInfo()
    col_algo_info = GemmAlgoInfo()

    def fill_algo_info(algo_info, line):
        algo_info.data_order = line.split()[2]
        algo_info.algo_id = int(line.split("Id=")[1].split(",")[0])
        algo_info.tile = int(line.split("tileIdx=")[1].split("(")[0])
        algo_info.splitk = int(line.split("splitK=")[1].split()[0])
        algo_info.reduc = int(line.split("reduc=")[1].split()[0])
        algo_info.swizzle = int(line.split("swizzle=")[1].split()[0])
        algo_info.custom = int(line.split("custom=")[1].split()[0])
        algo_info.stages = int(line.split("stages=")[1].split("}")[0])
        algo_info.workspace = int(line.split("workspace=")[1].split()[0])

    with open(log, "r") as fin:
        for line in fin:
            if line.startswith(">>>"):
                col32_algo_info.speedup = (
                    col32_algo_info.fp16_time / col32_algo_info.int8_time
                )
                col_algo_info.speedup = (
                    col_algo_info.fp16_time / col_algo_info.int8_time
                )
                if col32_algo_info.int8_time < col_algo_info.int8_time:
                    best_gemm_algos.append(deepcopy(col32_algo_info))
                else:
                    best_gemm_algos.append(deepcopy(col_algo_info))
            elif line.startswith("m "):
                shape = tuple([int(s.split()[1]) for s in line.split(";")])
                col32_algo_info.shape = shape
                col_algo_info.shape = shape
            elif line.startswith("Device"):
                sm = int(line.split("SM")[1].strip()[:-1])
                col32_algo_info.sm = sm
                col_algo_info.sm = sm
            elif line.startswith("FP16 NN-gemm"):
                fp16_time = float(line.split("exec_time")[1].strip()[:-4])
                col32_algo_info.fp16_time = fp16_time
                col_algo_info.fp16_time = fp16_time
            elif line.startswith("INT8 NT-gemm with"):
                col32_time = float(line.split("exec_time")[1].strip()[:-4])
                col32_algo_info.int8_time = col32_time
            elif line.startswith("INT8 TN-gemm with"):
                col_time = float(line.split("exec_time")[1].strip()[:-4])
                col_algo_info.int8_time = col_time
            elif line.startswith("INT8 NT-gemm"):
                fill_algo_info(col32_algo_info, line)
            elif line.startswith("INT8 TN-gemm"):
                fill_algo_info(col_algo_info, line)

    return best_gemm_algos


def search(mnk_set, output_cfg_file, output_cfg_str):
    tmp_output_file = "tmp_output.log"
    tmp_shell = "tmp_gemm_test.sh"

    rm(tmp_output_file)
    rm(tmp_shell)

    # compile the gemm_test library
    os.system("nvcc -o gemm gemm.cpp -lcublasLt -lcublas")

    with open(tmp_shell, "w") as fin:
        for m, n, k in mnk_set:
            fin.write("./gemm {} {} {} >> {}\n".format(m, n, k, tmp_output_file))
    print("Start searching...")
    os.system("sh {} > {}".format(tmp_shell, tmp_output_file))

    best_gemm_algos = extract(tmp_output_file)
    for d in best_gemm_algos:
        d_str = "{:>5d} {:>5d} {:>5d}   {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>7d}   {:.4f} {:.4f} {:.2f}   {:>2d} {}".format(
            d.shape[0],
            d.shape[1],
            d.shape[2],
            d.algo_id,
            d.tile,
            d.splitk,
            d.reduc,
            d.swizzle,
            d.custom,
            d.stages,
            d.workspace,
            d.fp16_time,
            d.int8_time,
            d.speedup,
            d.sm,
            d.data_order,
        )
        output_cfg_str.append((d.shape[0], d.shape[1], d.shape[2], d_str))

    def cmp(x, y):
        if x[2] != y[2]:
            return sign(x[2] - y[2])
        elif x[1] != y[1]:
            return sign(x[1] - y[1])
        else:
            return sign(x[0] - y[0])

    output_cfg_str.sort(key=functools.cmp_to_key(cmp))

    with open(output_cfg_file, "w") as fout:
        for s in output_cfg_str:
            fout.write(s[3] + "\n")

    rm(tmp_output_file)
    rm(tmp_shell)


def gemm_test(hidden_dim, inner_dim, vocab_size, min_bsz, max_bsz):
    sm = get_sm()
    if sm < 75:
        raise RuntimeError("int8 gemm is only supported on GPUs with SM >= 75.")

    # All (m, n, k) which may be searched.
    mnk_set = set()
    for bsz in range(min_bsz, max_bsz + 1):
        m = bsz if bsz < BORDER else ((bsz + STRIDE - 1) // STRIDE) * STRIDE
        if hidden_dim is not None and inner_dim is not None:
            nk = base_nk(hidden_dim, inner_dim)
            for n, k in nk:
                mnk_set.add((m, n, k))
        if hidden_dim is not None and vocab_size is not None:
            mnk_set.add((m, vocab_size, hidden_dim))
        elif hidden_dim is not None:
            pass

    # Existing (m, n, k).
    dir_name = "configs"
    mkdir(dir_name)
    output_cfg_file = "{}/igemm_sm{}.cfg".format(dir_name, sm)
    exist_mnk_set = set()
    output_cfg_str = []
    if os.path.exists(output_cfg_file):
        with open(output_cfg_file, "r") as fin:
            for line in fin:
                m, n, k = [int(x) for x in line.split()[:3]]
                exist_mnk_set.add((m, n, k))
                output_cfg_str.append((m, n, k, line.rstrip()))

    # (m, n, k) to be searched.
    mnk_set -= exist_mnk_set
    if len(mnk_set) <= 0:
        print("No gemm shapes need to be searched.")
        return
    search(mnk_set, output_cfg_file, output_cfg_str)


def check_args(args):
    assert (
        args.hidden_dim is not None
        and (args.inner_dim is not None or args.vocab_size is not None)
        and args.hidden_dim > 0
        and (args.inner_dim is None or args.inner_dim > 0)
        and (args.vocab_size is None or args.vocab_size > 0)
        and 1 <= args.min_bsz <= args.max_bsz
    )
    if args.min_bsz > BORDER:
        args.min_bsz = (args.min_bsz // STRIDE) * STRIDE
        print("Adjust the min_bsz to {}.".format(args.min_bsz))
    if args.max_bsz > BORDER:
        args.max_bsz = ((args.max_bsz + STRIDE - 1) // STRIDE) * STRIDE
        print("Adjust the max_bsz to {}.".format(args.max_bsz))


def parse_args():
    parser = argparse.ArgumentParser(
        description="search for the best int8 gemm algorithm",
        usage="python gemm_test.py -hd 1024 -id 4096 -v 32000 -minb 1 -maxb 100",
    )
    parser.add_argument(
        "--hidden_dim",
        "-hd",
        type=int,
        help="hidden dimension of the model",
    )
    parser.add_argument(
        "--inner_dim",
        "-id",
        type=int,
        help="inner dimension of the ffn layer",
    )
    parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        help="vocabulary size of the model",
    )
    parser.add_argument(
        "--min_bsz",
        "-minb",
        type=int,
        default=1,
        help="minimal batch token size",
    )
    parser.add_argument(
        "--max_bsz",
        "-maxb",
        type=int,
        default=MAX_BSZ,
        help="maximal batch token size",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    check_args(args)

    gemm_test(
        args.hidden_dim, args.inner_dim, args.vocab_size, args.min_bsz, args.max_bsz
    )
