import argparse
import os
from dataclasses import dataclass, asdict
from copy import deepcopy

FLOAT_MAX = float(1e9)
INTERVAL = 32
BORDER = 512
MAX_BSZ = 20000
COMMON_SHAPE = [(512, 2048), (768, 3072), (1024, 4096)]
SM = [75, 80]


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


def search(hidden_dim, inner_dim, vocab_size, min_bsz, max_bsz, nk_set, sm):
    dir_name = "configs_sm{}".format(sm)
    tmp_output_file = "{}/tmp_output.log".format(dir_name)
    tmp_shell = "{}/tmp_gemm_test.sh".format(dir_name)
    if len(nk_set) == 4:
        output_cfg_file = "{}/h{}_i{}_b{}-{}.cfg".format(
            dir_name, hidden_dim, inner_dim, min_bsz, max_bsz
        )
    elif len(nk_set) == 1:
        output_cfg_file = "{}/h{}_v{}_b{}-{}.cfg".format(
            dir_name, hidden_dim, vocab_size, min_bsz, max_bsz
        )
    else:
        raise ValueError("Wrong gemm shapes to be searched.")
    print("Search best gemm algorithms for shapes (m, n, k):")
    for shape in nk_set:
        print("  - (m, {}, {})".format(shape[0], shape[1]))
    print("where m is in the range [{}, {}].\n".format(min_bsz, max_bsz))

    mkdir(dir_name)
    rm(tmp_output_file)
    rm(tmp_shell)
    if os.path.exists(output_cfg_file):
        print(
            "The best config of the gemm shapes to be searched already exists ({}).".format(
                output_cfg_file
            )
        )
        return

    # compile the gemm_test library
    os.system("nvcc -o gemm gemm.cpp -lcublasLt -lcublas")

    with open(tmp_shell, "w") as fin:
        for n, k in nk_set:
            for bsz in range(min_bsz, min(max_bsz, BORDER), 1):
                fin.write("./gemm {} {} {} >> {}\n".format(bsz, n, k, tmp_output_file))
            for bsz in range(BORDER, max_bsz + 1, INTERVAL):
                fin.write("./gemm {} {} {} >> {}\n".format(bsz, n, k, tmp_output_file))
    print("Start searching...")
    os.system("sh {} > {}".format(tmp_shell, tmp_output_file))

    best_gemm_algos = extract(tmp_output_file)
    best_gemm_algos_dict = [asdict(x) for x in best_gemm_algos]
    with open(output_cfg_file, "w") as fout:
        for d in best_gemm_algos_dict:
            fout.write(
                "{:>5d} {:>4d} {:>4d}   {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>2d} {:>7d}   {:.4f} {:.4f} {:.2f}   {:>2d} {}\n".format(
                    d["shape"][0],
                    d["shape"][1],
                    d["shape"][2],
                    d["algo_id"],
                    d["tile"],
                    d["splitk"],
                    d["reduc"],
                    d["swizzle"],
                    d["custom"],
                    d["stages"],
                    d["workspace"],
                    d["fp16_time"],
                    d["int8_time"],
                    d["speedup"],
                    d["sm"],
                    d["data_order"],
                )
            )

    rm(tmp_output_file)
    rm(tmp_shell)


def gemm_test(hidden_dim, inner_dim, vocab_size, min_bsz, max_bsz):
    sm = get_sm()

    layer_nk_set = []
    logit_nk_set = []
    if (
        hidden_dim is not None
        and inner_dim is not None
        and ((hidden_dim, inner_dim) not in COMMON_SHAPE or sm not in SM)
    ):
        layer_nk_set = base_nk(hidden_dim, inner_dim)
    if hidden_dim is not None and vocab_size is not None:
        logit_nk_set.append((vocab_size, hidden_dim))
    if len(layer_nk_set) <= 0 and len(logit_nk_set) <= 0:
        print("No gemm shapes need to be searched.")
        return

    if len(layer_nk_set) > 0:
        search(hidden_dim, inner_dim, vocab_size, min_bsz, max_bsz, layer_nk_set, sm)

    if len(logit_nk_set) > 0:
        search(hidden_dim, inner_dim, vocab_size, min_bsz, max_bsz, logit_nk_set, sm)


def check_args(args):
    assert (
        args.hidden_dim is not None
        and (args.inner_dim is not None or args.vocab_size is not None)
        and args.hidden_dim > 0
        and (args.inner_dim is None or args.inner_dim > 0)
        and (args.vocab_size is None or args.vocab_size > 0)
        and 1 <= args.min_bsz <= args.max_bsz <= MAX_BSZ
    )
    if args.min_bsz > BORDER:
        args.min_bsz = (args.min_bsz // INTERVAL) * INTERVAL
        print("Adjust the min_bsz to {}.".format(args.min_bsz))
    if args.max_bsz > BORDER:
        args.max_bsz = ((args.max_bsz + INTERVAL - 1) // INTERVAL) * INTERVAL
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
