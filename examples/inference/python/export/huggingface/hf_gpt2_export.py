"""
Export Hugging Face GPT2 models to hdf5 format.
"""
import os
import sys

sys.path.insert(0, "/mlx_devbox/users/zhoubofan/playground/")
sys.path.insert(0, "/mlx_devbox/users/zhoubofan/playground/lightseq")
import h5py
import numpy as np
from collections import OrderedDict
from transformers import GPT2LMHeadModel
from lightseq.training.ops.pytorch.export import fill_hdf5_layer
from export.util import parse_args
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_transformation_var_names(n_layers, hf_prefix="transformer."):
    mapping = {}
    mapping_with_transpose = {}

    mapping["model.embedding.position_embedding"] = hf_prefix + "wpe.weight"
    mapping["model.embedding.weight"] = hf_prefix + "wte.weight"
    for i in range(n_layers):
        selfattn_prefix = f"model.decoder.layers.{i}.selfatt_layer."
        ffn_prefix = f"model.decoder.layers.{i}.ffn_layer."
        mapping[selfattn_prefix + "norm_layer.weight"] = (
            hf_prefix + f"h.{i}.ln_1.weight"
        )
        mapping[selfattn_prefix + "norm_layer.bias"] = hf_prefix + f"h.{i}.ln_1.bias"
        mapping[selfattn_prefix + "selfatt.qkv_transform.weight"] = (
            hf_prefix + f"h.{i}.attn.c_attn.weight"
        )
        mapping[selfattn_prefix + "selfatt.qkv_transform.bias"] = (
            hf_prefix + f"h.{i}.attn.c_attn.bias"
        )
        mapping[selfattn_prefix + "selfatt.output_transform.weight"] = (
            hf_prefix + f"h.{i}.attn.c_proj.weight"
        )
        mapping[selfattn_prefix + "selfatt.output_transform.bias"] = (
            hf_prefix + f"h.{i}.attn.c_proj.bias"
        )

        mapping_with_transpose[ffn_prefix + "dense1.weight"] = (
            hf_prefix + f"h.{i}.mlp.c_fc.weight"
        )
        mapping_with_transpose[ffn_prefix + "dense1.bias"] = (
            hf_prefix + f"h.{i}.mlp.c_fc.bias"
        )
        mapping_with_transpose[ffn_prefix + "dense2.weight"] = (
            hf_prefix + f"h.{i}.mlp.c_proj.weight"
        )
        mapping_with_transpose[ffn_prefix + "dense2.bias"] = (
            hf_prefix + f"h.{i}.mlp.c_proj.bias"
        )
        mapping[ffn_prefix + "norm_layer.weight"] = hf_prefix + f"h.{i}.ln_2.weight"
        mapping[ffn_prefix + "norm_layer.bias"] = hf_prefix + f"h.{i}.ln_2.bias"
    mapping["model.decoder.output_norm.weight"] = hf_prefix + "ln_f.weight"
    mapping["model.decoder.output_norm.bias"] = hf_prefix + "ln_f.bias"

    return mapping, mapping_with_transpose


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "ln_1 weight",
        "multihead_norm_bias": "ln_1 bias",
        # GPT2's Conv1D don't need transpose
        # https://github.com/huggingface/transformers/blob/9ec0f01b6c3aff4636869aee735859fb6f89aa98/src/transformers/modeling_utils.py#L1400
        "multihead_project_kernel_qkv": "attn c_attn weight",
        "multihead_project_bias_qkv": "attn c_attn bias",
        "multihead_project_kernel_output": "attn c_proj weight",
        "multihead_project_bias_output": "attn c_proj bias",
        "ffn_norm_scale": "ln_2 weight",
        "ffn_norm_bias": "ln_2 bias",
        "ffn_first_kernel": "mlp c_fc weight",
        "ffn_first_bias": "mlp c_fc bias",
        "ffn_second_kernel": "mlp c_proj weight",
        "ffn_second_bias": "mlp c_proj bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "ln_f weight",
        "norm_bias": "ln_f bias",
        "token_embedding": "wte",
        # manually process position_embedding to customize for max_step
        # "position_embedding": "wpe",
    }
)


def extract_gpt_weights(
    output_file,
    model_dir,
    generation_method,
    beam_size=1,
    length_penalty=1.0,
    topk=1,
    topp=0.75,
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id=69764,
    pad_id=69765,
    max_step=255,
    extra_decode_length=0,
):

    # load var names
    model = torch.load(
        model_dir
    )  # bytedseq.pyneurst.models.gpt2.Gpt2.from_pretrained(model_dir)
    print(model.keys())
    print("-------")
    print(model["global_step"])
    print("-------")
    print(model["model_configs"])
    print("-------")
    # print(model['state_dict'].keys())
    # print('-------')
    config = model["model_configs"]["task.params"]["model.params"]
    head_num = config["num_attention_heads"]

    encoder_state_dict = model["state_dict"]

    hfmp, hftmp = get_transformation_var_names(
        n_layers=config["num_layers"], hf_prefix="transformer."
    )

    trans_state_dict = OrderedDict()
    for k, v in encoder_state_dict.items():
        if k in hfmp.keys():
            trans_state_dict[hfmp[k]] = v
        elif k in hftmp.keys():
            if v.dim() == 1:
                trans_state_dict[hftmp[k]] = v
            else:
                # print(v.dim(), k)
                trans_state_dict[hftmp[k]] = v.transpose(0, 1)
        else:
            print("error")
            exit(-1)

    encoder_state_dict = trans_state_dict

    print(encoder_state_dict.keys())
    print("-------")

    enc_var_name_list = list(trans_state_dict.keys())

    # initialize output file
    output_file += ".hdf5"
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name_split[2].isdigit():
            continue
        layer_id = int(name_split[2])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        fill_hdf5_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # special handling for position embedding
    position_emb = encoder_state_dict["transformer.wpe.weight"]
    _max_allowed_step, _hidden_size = position_emb.shape
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
        "model_conf/length_penalty", data=length_penalty, dtype="f4"
    )
    hdf5_file.create_dataset(
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in generation_method]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/beam_size", data=beam_size, dtype="i4")
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/extra_decode_length", data=extra_decode_length, dtype="i4"
    )

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

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # head_num = model.config.n_head
    # encoder_state_dict = model.state_dict()
    # enc_var_name_list = list(encoder_state_dict.keys())
    # print(encoder_state_dict.keys())
    # exit(0)

    args = parse_args()
    # if args.generation_method not in ["topk", "topp", "ppl", "beam_search"]:
    args.generation_method = "topk"
    output_lightseq_model_name = "lightseq_gpt2_pyneurst"  # or "lightseq_gpt2_large"
    input_huggingface_gpt_model = (
        "/mlx_devbox/users/zhoubofan/playground/ckpt-500400.pt"  # or "gpt2-large"
    )
    topk = 1
    topp = 0.75
    beam_size = 4
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    length_penalty = 0.6
    eos_id = 69765
    pad_id = 69766
    max_step = 255
    extra_decode_length = 0  # use positive length to avtivate it
    extract_gpt_weights(
        output_lightseq_model_name,
        input_huggingface_gpt_model,
        generation_method=args.generation_method,
        length_penalty=length_penalty,
        beam_size=beam_size,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
        max_step=max_step,
        extra_decode_length=extra_decode_length,
    )
