import os
from collections import OrderedDict

import tensorflow as tf
from gpt_pb2 import Gpt
from utils import fill_layer
from transformers import GPT2LMHeadModel

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
        "position_embedding": "wpe",
    }
)


def extract_gpt_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    topk=1,
    topp=0.75,
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id=50256,
    pad_id=50257,
):
    gpt = Gpt()
    # load var names
    encoder_state_dict = GPT2LMHeadModel.from_pretrained(model_dir).state_dict()
    enc_var_name_list = list(encoder_state_dict.keys())

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name_split[2].isdigit():
            continue
        layer_id = int(name_split[2])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    for layer_id in sorted(enc_tensor_names.keys()):
        fill_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            gpt.encoder_stack.add(),
            enc_layer_mapping_dict,
        )

    # fill src_embedding
    fill_layer(
        enc_var_name_list,
        encoder_state_dict,
        gpt.src_embedding,
        src_emb_mapping_dict,
    )

    # fill in conf
    gpt.model_conf.head_num = head_num
    gpt.model_conf.src_padding_id = pad_id
    gpt.model_conf.sampling_method = generation_method
    gpt.model_conf.topp = topp
    gpt.model_conf.topk = topk
    gpt.model_conf.eos_id = eos_id

    print("Wrting to {0}".format(output_file))
    with tf.io.gfile.GFile(output_file, "wb") as fout:
        fout.write(gpt.SerializeToString())

    gpt = Gpt()
    with tf.io.gfile.GFile(output_file, "rb") as fin:
        gpt.ParseFromString(fin.read())
    print(gpt.model_conf)


if __name__ == "__main__":
    output_lightseq_model_name = "lightseq_gpt2.pb"
    input_huggingface_gpt_model = "gpt2"
    head_number = 12
    # generation_method should be "topk" or "topp"
    generation_method = "topk"
    topk = 1
    topp = 0.75
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id = 50256
    pad_id = 50257
    extract_gpt_weights(
        output_lightseq_model_name,
        input_huggingface_gpt_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
    )
