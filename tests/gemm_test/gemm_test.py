import argparse

from lightseq.training import gemm_test


def check_args(args):
    assert (
        args.hidden_dim is not None
        and (args.inner_dim is not None or args.vocab_size is not None)
        and args.hidden_dim > 0
        and (args.inner_dim is None or args.inner_dim > 0)
        and (args.vocab_size is None or args.vocab_size > 0)
        and 1 <= args.min_bsz <= args.max_bsz
    )


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
        default=10000,
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
