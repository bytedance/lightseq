import argparse
import subprocess
import os
import re

import h5py
import numpy as np

from export.huggingface.hf_bart_export import extract_transformer_weights
from export.huggingface.hf_gpt2_export import extract_gpt_weights
from export.huggingface.ls_hf_quant_gpt2_export import (
    extract_gpt_weights as extract_quant_gpt_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark for lightseq cpp inference")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--beam_size", type=int, default=1, help="beam size in beam search"
    )
    parser.add_argument(
        "--input_seq_len", type=int, default=32, help="input sequence length"
    )
    parser.add_argument(
        "--output_seq_len", type=int, default=32, help="output sequence length"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-base",
        help="huggingface pretrain model or checkpoint path",
    )
    parser.add_argument(
        "--sampling_method", type=str, default="beam_search", help="generation method"
    )
    parser.add_argument("--topk", type=int, default=4, help="topk when topk sampling")
    parser.add_argument(
        "--topp", type=float, default=0.75, help="topp when topp sampling"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="length penalty when generation",
    )
    parser.add_argument(
        "--enable_quant",
        type=bool,
        default=False,
        help="is quantized model",
    )
    args = parser.parse_args()
    return args


def generate_model(args):
    model_path = "lightseq_{}_bench.hdf5".format(args.model_name.split("/")[-1])
    if "bart" in args.model_name:
        if not os.path.exists(model_path):
            extract_transformer_weights(
                model_path.split(".")[0],
                args.model_name,
                generation_method=args.sampling_method,
                beam_size=args.beam_size,
                topk=args.topk,
                topp=args.topp,
                max_step=256 if args.output_seq_len < 256 else args.output_seq_len,
                extra_decode_length=args.output_seq_len - args.input_seq_len,
                only_decoder=False,
                length_penalty=args.length_penalty,
                save_proto=False,
            )

            return

        with h5py.File(model_path, "r+") as f:
            f["model_conf"]["sampling_method"][()] = np.array(
                [ord(c) for c in args.sampling_method]
            ).astype(np.int8)
            f["model_conf"]["beam_size"][()] = args.beam_size
            f["model_conf"]["topk"][()] = args.topk
            f["model_conf"]["topp"][()] = args.topp
            f["model_conf"]["extra_decode_length"][()] = (
                args.output_seq_len - args.input_seq_len
            )
            f["model_conf"]["length_penalty"][()] = args.length_penalty

    elif "gpt" in args.model_name or "clm" in args.model_name:
        if args.enable_quant:
            model_path = "lightseq_quant_gpt2_bench.hdf5".format(
                args.model_name.split("/")[-1]
            )

            extract_quant_gpt_weights(
                model_path,
                args.model_name,
                generation_method=args.sampling_method,
                topk=args.topk,
                topp=args.topp,
                eos_id=50256,
                pad_id=50257,
                max_step=args.input_seq_len + args.output_seq_len,
            )
        else:
            extract_gpt_weights(
                model_path.split(".")[0],
                args.model_name,
                generation_method=args.sampling_method,
                topk=args.topk,
                topp=args.topp,
                eos_id=50256,
                pad_id=50257,
                max_step=args.input_seq_len + args.output_seq_len,
            )
    else:
        raise ValueError(
            "can't infer model type from {}, generate model failed".format(
                args.model_name
            )
        )

    return os.path.realpath(model_path)


def run_bench(model_path, args):
    bin_dir = os.path.realpath("../../../build/")
    print(bin_dir)
    bin_path = "examples/inference/cpp/transformer_example"
    full_bin_path = os.path.join(bin_dir, bin_path)
    if not os.path.exists(full_bin_path):
        raise RuntimeError(
            "can't find {}, please check if you build inference cpp example successfully".format(
                full_bin_path
            )
        )
    res = subprocess.check_output(
        " ".join(
            [
                full_bin_path,
                model_path,
                str(args.batch_size),
                str(args.input_seq_len),
            ]
        ),
        # check=True,
        # capture_output=True,
        shell=True,
        # text=True,
    )
    latency = 0
    print(res)
    for line in res.stdout.splitlines():
        if "latency" in line:
            latency = re.findall(r"\d+\.\d+ms", line)[0]
    return latency


def main():
    args = parse_args()
    model_path = generate_model(args)


if __name__ == "__main__":
    main()
