import argparse
from re import L
import os
import h5py
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="export fairseq checkpoint", usage="")
    parser.add_argument(
        "--model_file",
        "-m",
        type=str,
        required=True,
        help="path of pytorch model repo",
    )
    parser.add_argument(
        "--generation_method",
        "-g",
        type=str,
        required=True,
        choices=["beam_search", "topk_greedy", "topk", "topp", "ppl"],
        help="generation method",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--topp",
        type=float,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--extra_decode_length",
        type=int,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def check_arguements(args):
    if args.generation_method == "beam_search":
        if args.beam_size == None:
            raise Exception("set 'beam_size' value while using beam search method.")
    elif args.generation_method == "topk":
        if args.topk == None:
            raise Exception("set 'topk' value while using topk sample method.")
        ...
    elif args.generation_method == "topp":
        if args.topp == None:
            raise Exception("set 'topp' value while using topp sample method.")

    if args.eos_id == None:
        raise Exception("eos id should not be set as None")

    if args.padding_id == None:
        raise Exception("padding id shoud not be set as None")

    if args.beam_size == None:
        args.beam_size = 1

    if args.topp == None:
        args.topp = 1.0

    if args.topk == None:
        args.topk = 1

    return True


class ModelArguements(object):
    def __init__(self, args):
        self.model_file = os.path.abspath(args.model_file)
        if not os.path.isfile(self.model_file):
            raise Exception(f"there is no such model file {self.model_file}")

        self.model_repo = os.path.dirname(self.model_file)
        self.generation_method = args.generation_method
        self.beam_size = args.beam_size
        self.topk = args.topk
        self.topp = args.topp
        self.eos_id = None
        self.bos_id = None
        self.config_path = os.path.join(self.model_repo, "config.json")

        if not os.path.isfile(self.config_path):
            raise Exception(f"there is no such config file {self.config_path}")

        config_file = open(self.config_path)
        config = json.load(config_file)
        config_file.close()

        self.padding_id = config.get("pad_token_id")
        self.max_step = config.get("max_sequence_length")
        self.hidden_size = config.get("hidden_size")
        self.inner_size = config.get("intermediate_size")
        self.head_num = config.get("num_attention_heads")
        self.vocab_size = config.get("vocab_size")
        self.layer_num = config.get("num_hidden_layers")
        self.extra_decode_length = (
            self.max_step
            if args.extra_decode_length is None
            else args.extra_decode_length
        )


def apply_rule(proto_name, ckpt_rule, tensor_names, state_dict):
    def check_rule(tensor_name, rule):
        if "Adam" in tensor_name or "adam" in tensor_name:
            return False
        assert isinstance(rule, str) and rule
        rule = rule.split("-")
        assert len(rule) < 3
        if len(rule) == 2:
            white, black = rule[0].split(" "), rule[1].split(" ")
        else:
            white, black = rule[0].split(" "), []
        for b in black:
            if b in tensor_name.split("."):
                return False
        for w in white:
            if w not in tensor_name.split("."):
                return False
        return True

    expression = [ele for ele in ckpt_rule.split("&&") if ele.startswith("expression_")]

    ckpt_rule = [
        ele for ele in ckpt_rule.split("&&") if not ele.startswith("expression_")
    ]

    assert (len(ckpt_rule) > 0 and len(expression) < 2) or (
        len(ckpt_rule) == 0 and len(expression) > 0
    )

    if len(expression) < 2:
        expression = "" if not expression else expression[0].split("_")[1]
    else:
        expression = [exp.split("_")[1] for exp in expression]

    target_tn = []
    for cr in ckpt_rule:
        tmp = []
        for tn in tensor_names:
            if check_rule(tn, cr):
                tmp.append(tn)
        assert len(tmp) == 1
        target_tn.extend(tmp)
    target_tensor = [state_dict[name].float() for name in target_tn]
    tt = {}
    if target_tensor:
        exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)
    else:
        if not isinstance(expression, list):
            expression = [expression]
        exec("tt['save'] = [%s]" % ",".join(expression))

    try:
        target_tensor = np.concatenate(tt["save"], axis=-1)
    except:
        target_tensor = tt["save"]
    print(
        "%s -> %s, convert finished!"
        % (target_tn if target_tn else "created", proto_name)
    )
    return target_tensor[0] if type(target_tensor) is list else target_tensor


def fill_hdf5_layer(
    tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix, mapping_dict
):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        hdf5_file.create_dataset(
            hdf5_dataset_prefix + proto_name, data=target_tensor.flatten().tolist()
        )
