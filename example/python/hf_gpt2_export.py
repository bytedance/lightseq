import os
from collections import OrderedDict

import tensorflow as tf
from gpt_pb2 import Gpt
from utils import fill_layer
from transformers import GPT2LMHeadModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


""" key是proto参数的值，value是一个强大的表达式，每个&&分割tensor name的匹配路径或表达式，每个匹配
路径的子pattern用空格分隔，表达式用expression_开头，可以对每个tensor进行单独操作，支持多个表达式。多个匹配路径
和表达式最后会concat，axis=-1 """
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "ln_1 weight",
        "multihead_norm_bias": "ln_1 bias",
        # GPT2的Conv1D无需额外transpose https://github.com/huggingface/transformers/blob/9ec0f01b6c3aff4636869aee735859fb6f89aa98/src/transformers/modeling_utils.py#L1400
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
):
    gpt = Gpt()
    # load var names
    encoder_state_dict = GPT2LMHeadModel.from_pretrained(model_dir).state_dict()
    enc_var_name_list = list(encoder_state_dict.keys())
    # import pdb;pdb.set_trace()

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
    gpt.model_conf.src_padding_id = 1
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
    input_huggingface_gpt_model = ("gpt2")
    head_number = 12
    # in order to get score, we should use `beam_search` inference method
    generation_method = "beam_search"
    topk = 1
    topp = 0.75
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id = 50256
    extract_gpt_weights(
        output_lightseq_model_name,
        input_huggingface_gpt_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
    )
