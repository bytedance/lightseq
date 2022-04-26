"""
Export Hugging Face quantized GPT2 models to hdf5 format.
"""
import os
import h5py
from collections import OrderedDict

import numpy as np
import torch
from lightseq.training.ops.pytorch.export import apply_rule
from lightseq.training.ops.pytorch.export_quant import quantize
from export.util import parse_args

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "self_attn_layer_norm weight",
        "multihead_norm_bias": "self_attn_layer_norm bias",
        "multihead_project_kernel_qkv": "self_attn qkv_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn qkv_proj bias",
        "multihead_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
        # weight_clip_max
        "multihead_project_kernel_qkv_clip_max": "self_attn qkv_proj weight_quant clip_value_max",
        "multihead_project_kernel_output_clip_max": "self_attn out_proj weight_quant clip_value_max",
        "ffn_first_kernel_clip_max": "fc1 weight_quant clip_value_max",
        "ffn_second_kernel_clip_max": "fc2 weight_quant clip_value_max",
        # act_clip_max
        "multihead_ln_clip_max": "self_attn qkv_proj input_quant clip_value_max",
        "multihead_project_output_clip_max": "self_attn out_proj input_quant clip_value_max",
        "ffn_ln_clip_max": "fc1 input_quant clip_value_max",
        "ffn_first_act_clip_max": "fc2 input_quant clip_value_max",
        "multihead_qkv_dense_clip_max": "self_attn qkv_proj output_quant clip_value_max",
        "multihead_output_dense_clip_max": "self_attn out_proj output_quant clip_value_max",
        "ffn_first_output_clip_max": "fc1 output_quant clip_value_max",
        "self_qkv_bias_out_clip_max": "self_attn attention_quant clip_value_max",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "ln_f weight",
        "norm_bias": "ln_f bias",
        "output_ln_clip_max": "lm_head input_quant clip_value_max",
        "logits_clip_max": "lm_head output_quant clip_value_max",
    }
)


def fill_quant_hdf5_layer(
    tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix, mapping_dict
):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        if proto_name.endswith("_clip_max"):
            hdf5_file.create_dataset(
                hdf5_dataset_prefix + proto_name, data=float(target_tensor[0])
            )
        else:
            hdf5_file.create_dataset(
                hdf5_dataset_prefix + proto_name,
                data=target_tensor,
            )


def extract_gpt_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    topk=1,
    topp=0.75,
    eos_id=50256,
    pad_id=50257,
    max_step=50,
):
    # load var names
    state_dict = torch.load(model_dir, "cpu")

    var_name_list = list(state_dict.keys())

    for name in var_name_list:
        if name.endswith("weight_quant.clip.clip_value_max"):
            state_dict[name[:-26]] = torch.Tensor(
                quantize(state_dict[name[:-26]].numpy(), 127, state_dict[name].numpy())
            ).to(torch.uint8)

    # initialize output file
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name_split[2].isdigit():
            continue
        layer_id = int(name_split[2])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        fill_quant_hdf5_layer(
            enc_tensor_names[layer_id],
            state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_quant_hdf5_layer(
        var_name_list,
        state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # handling token_embeddings for GPT
    token_embedding = state_dict["transformer.wte.weight"]
    token_embedding = quantize(
        token_embedding.numpy(),
        127,
        state_dict["transformer.wte.emb_quant.clip.clip_value_max"].numpy(),
    ).transpose()
    print(f"processed token_embedding, shape: {token_embedding.shape}")
    hdf5_file.create_dataset(
        "src_embedding/token_embedding", data=token_embedding, dtype="uint8"
    )
    hdf5_file.create_dataset(
        "src_embedding/emb_clip_max",
        data=state_dict["transformer.wte.emb_quant.clip.clip_value_max"],
    )

    # special handling for position embedding
    position_emb = state_dict["transformer.wpe.weight"]
    _max_allowed_step, _ = position_emb.shape
    if max_step > _max_allowed_step:
        print(f"max_step {max_step} exceed max allowed step, abort.")
        return
    # truncate position embedding for max_step
    position_emb = position_emb[:max_step, :]
    print(
        f"processed position_embedding with max_step constriant, shape: {position_emb.shape}"
    )
    position_emb = position_emb.flatten().tolist()
    hdf5_file.create_dataset(
        "src_embedding/position_embedding", data=position_emb, dtype="f4"
    )

    # save number of layers metadata
    hdf5_file.create_dataset(
        "model_conf/n_encoder_stack", data=len(enc_tensor_names), dtype="i4"
    )
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/src_padding_id", data=pad_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in generation_method]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        if key == "sampling_method":
            value = "".join(map(chr, value[()]))
        else:
            value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    args = parse_args()
    assert args.generation_method in ["topk", "topp", "ppl"]
    model_name = ".".join(args.model.split(".")[:-1])
    hdf5_path = f"{model_name}.hdf5"

    head_number = 12  # 20 for "gpt2-large"
    topk = 1
    topp = 0.75
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id = 50256
    pad_id = 50257
    max_step = 50
    extract_gpt_weights(
        hdf5_path,
        args.model,
        head_num=head_number,  # layer number
        generation_method=args.generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
        max_step=max_step,
    )
