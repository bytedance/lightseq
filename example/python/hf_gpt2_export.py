import os
import math
from collections import OrderedDict

import numpy as np
import torch
import tensorflow as tf
from gpt_pb2 import Gpt
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


def fill_layer(tensor_names, stete_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        expression = [
            ele for ele in ckpt_rule.split("&&") if ele.startswith("expression_")
        ]

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
            if len(tmp) != 1:
                print(tmp, cr)
            assert len(tmp) == 1
            target_tn.extend(tmp)
        target_tensor = [stete_dict[name] for name in target_tn]
        tt = {}
        if target_tensor:
            exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)
        else:
            if not isinstance(expression, list):
                expression = [expression]
            exec("tt['save'] = [%s]" % ",".join(expression))

        target_tensor = np.concatenate(tt["save"], axis=-1)
        print(
            "%s -> %s, shape: %s, convert finished."
            % (target_tn if target_tn else "created", proto_name, target_tensor.shape)
        )
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "encoder_attn {0} k_proj weight&&encoder_attn {0} v_proj weight".format(
            ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "encoder_attn {0} k_proj bias&&encoder_attn {0} v_proj bias".format(
            ele)
        for ele in range(dec_layer_num)
    ]

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
    }


def _get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    with tf.device("/cpu:0"):
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = hidden_size // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -
            log_timescale_increment
        )
        scaled_time = tf.expand_dims(
            position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat(
            [tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    return signal


def _gather_token_embedding(tensor_names, name2var_dict, tn_pattern, lang="en"):
    """ use pattern to diff source and target. """
    target_tn = []
    # lang_embedding = None
    for tn in tensor_names:
        if (tn_pattern in tn.split(".")) and ("weight" in tn.split(".")):
            target_tn.append(tn)
            continue
        # if tn == "lang_embeddings.weight":
        #     lang_embedding = name2var_dict[tn].numpy()
    # target_tn = sorted(target_tn, key=lambda x: int(x.split('_')[-1]))
    # print(target_tn)
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    # target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    # print(
    #     "lang embedding shape: {}, added {} embedding to token embeddings".format(
    #         lang, lang_embedding.shape
    #     )
    # )
    # target_tensor += lang_embedding[LANG2ID[lang]]
    print("token embedding shape is {}".format(target_tensor.shape))
    # print("token embedding shape is %s" % target_tensor.shape)

    return target_tensor


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
