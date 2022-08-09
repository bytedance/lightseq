import argparse

from lightseq.training import gemm_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="search for the best int8 gemm algorithm",
        usage="python gemm_test.py -hd 1024 -id 4096 -v 32000 -minb 1 -maxb 10 -d configs",
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
        default=10000,
        help="maximal batch token size",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default="/tmp/igemm_configs",
        help="path of the saved configs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    gemm_test(
        args.hidden_dim,
        args.inner_dim,
        args.vocab_size,
        args.min_bsz,
        args.max_bsz,
        args.dir,
    )
