"""
Export Hugging Face XGLM models to hdf5 format.
"""
import os
import h5py
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, XGLMForCausalLM
from lightseq.training.ops.pytorch.export import fill_hdf5_layer
import math

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
        "multihead_project_kernel_qkv": "self_attn q_proj weight&&self_attn k_proj weight&&self_attn v_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn q_proj bias&&self_attn k_proj bias&&self_attn v_proj bias",
        "multihead_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
    }
)


src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "layer_norm weight",
        "norm_bias": "layer_norm bias",
        # "token_embedding": "embed_tokens",
    }
)


def extract_xglm_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    topk=10,
    topp=0.75,
    eos_id=50256,
    pad_id=50257,
    max_step=100,
    extra_decode_length=0,
):
    # load var names
    encoder_state_dict = XGLMForCausalLM.from_pretrained(model_dir).state_dict()
    enc_var_name_list = list(encoder_state_dict.keys())

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

    token_emb = encoder_state_dict["model.embed_tokens.weight"]

    # scale embedding
    scale = math.sqrt(token_emb.shape[1])
    token_embedding = (token_emb.flatten() * scale).tolist()

    hdf5_file.create_dataset(
        "src_embedding/token_embedding", data=token_embedding, dtype="f4"
    )

    # special handling for position embedding
    position_emb = encoder_state_dict["model.embed_positions.weights"]
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
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in generation_method]).astype(np.int8),
        dtype="i1",
    )
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
    output_lightseq_model_name = "lightseq_incoder_base"
    input_huggingface_xglm_model = "facebook/incoder-1B"
    head_number = 32
    # generation_method should be "topk" or "topp"
    generation_method = "topk"
    topk = 1
    topp = 0.75

    eos_id = 2
    pad_id = 1
    max_step = 100
    extra_decode_length = 0  # use positive length to avtivate it
    extract_xglm_weights(
        output_lightseq_model_name,
        input_huggingface_xglm_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
        max_step=max_step,
        extra_decode_length=extra_decode_length,
    )