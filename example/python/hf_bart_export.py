import os
import math
from collections import OrderedDict

import numpy as np
import torch
import tensorflow as tf
from transformer_pb2 import Transformer
from transformers import BartForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


""" key是proto参数的值，value是一个强大的表达式，每个&&分割tensor name的匹配路径或表达式，每个匹配
路径的子pattern用空格分隔，表达式用expression_开头，可以对每个tensor进行单独操作，支持多个表达式。多个匹配路径
和表达式最后会concat，axis=-1 """
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

dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "self_attn_layer_norm weight",
        "self_norm_bias": "self_attn_layer_norm bias",
        "self_project_kernel_qkv": "self_attn q_proj weight&&self_attn k_proj weight&&self_attn v_proj weight&&expression_.transpose(0, 1)",
        "self_project_bias_qkv": "self_attn q_proj bias&&self_attn k_proj bias&&self_attn v_proj bias",
        "self_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "self_project_bias_output": "self_attn out_proj bias",
        "encdec_norm_scale": "encoder_attn_layer_norm weight",
        "encdec_norm_bias": "encoder_attn_layer_norm bias",
        "encdec_project_kernel_q": "encoder_attn q_proj weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_q": "encoder_attn q_proj bias",
        "encdec_project_kernel_output": "encoder_attn out_proj weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_output": "encoder_attn out_proj bias",
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
        "norm_scale": "layernorm_embedding weight",
        "norm_bias": "layernorm_embedding bias",
    }
)

trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "layernorm_embedding weight",
        "norm_bias": "layernorm_embedding bias",
        "shared_bias": "final_logits_bias",
    }
)

# shared_trg_emb_mapping_dict = OrderedDict(
#     {
#         "norm_scale": "layer_norm_emb weight",
#         "norm_bias": "layer_norm_emb bias",
#         "shared_bias": "pred_layer bias",
#     }
# )


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
        "encoder_attn {0} k_proj weight&&encoder_attn {0} v_proj weight".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "encoder_attn {0} k_proj bias&&encoder_attn {0} v_proj bias".format(ele)
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
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
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


def extract_transformer_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    max_step,
    extra_decode_length=50,
    beam_size=4,
    length_penalty: float = 0,
    topk=1,
    topp=0.75,
    lang="en",
    only_decoder=True,
):
    transformer = Transformer()
    # load var names
    reloaded = BartForConditionalGeneration.from_pretrained(model_dir).state_dict()
    # trg_emb_mapping_dict["shared_bias"] = (
    #     f"expression_np.zeros({reloaded['model.shared.weight'].numpy().shape[0]})"
    # )
    # trg_emb_mapping_dict["shared_bias"] = (reloaded['final_logits_bias'][0].numpy())

    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in reloaded:
        if k.startswith("model.encoder."):
            encoder_state_dict[k] = reloaded[k]
        if k.startswith("model.decoder."):
            decoder_state_dict[k] = reloaded[k]
        if k == "model.shared.weight":
            encoder_state_dict[k] = reloaded[k]
            decoder_state_dict[k] = reloaded[k]
        if k == "final_logits_bias":
            decoder_state_dict[k] = reloaded[k]

    dec_var_name_list = list(decoder_state_dict.keys())
    enc_var_name_list = list(encoder_state_dict.keys())
    # import pdb;pdb.set_trace()

    # fill each encoder layer's params
    if not only_decoder:
        enc_tensor_names = {}
        for name in enc_var_name_list:
            name_split = name.split(".")
            if len(name_split) <= 3 or not name_split[3].isdigit():
                continue
            layer_id = int(name_split[3])
            enc_tensor_names.setdefault(layer_id, []).append(name)

        for layer_id in sorted(enc_tensor_names.keys()):
            fill_layer(
                enc_tensor_names[layer_id],
                encoder_state_dict,
                transformer.encoder_stack.add(),
                enc_layer_mapping_dict,
            )

    # fill each decoder layer's params
    dec_tensor_names = {}
    for name in dec_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 3 or not name.split(".")[3].isdigit():
            continue
        layer_id = int(name.split(".")[3])
        dec_tensor_names.setdefault(layer_id, []).append(name)

    for layer_id in sorted(dec_tensor_names.keys()):
        fill_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill src_embedding
    if not only_decoder:
        fill_layer(
            enc_var_name_list,
            encoder_state_dict,
            transformer.src_embedding,
            src_emb_mapping_dict,
        )
        # bart position index starts from 2
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/configuration_bart.py#L208
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py#L821
        pos_emb_list = (
            encoder_state_dict["model.encoder.embed_positions.weight"]
            .numpy()[
                2 : 2 + max_step, :
            ]  # because in huggingface bart, the position embedding starts from 2
            .reshape([-1])
            .tolist()
        )
        transformer.src_embedding.position_embedding[:] = pos_emb_list
        print(
            "model.encoder.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
                encoder_state_dict["model.encoder.embed_positions.weight"]
                .numpy()[2 : 2 + max_step, :]
                .shape
            )
        )
        src_tb = _gather_token_embedding(
            enc_var_name_list, encoder_state_dict, "shared"
        )
        transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    trg_emb_mapping_dict.update(encode_output_mapping_dict)
    fill_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        trg_emb_mapping_dict,
    )
    pos_emb_list = (
        decoder_state_dict["model.decoder.embed_positions.weight"]
        .numpy()[2 : 2 + max_step, :]
        .reshape([-1])
        .tolist()
    )
    transformer.trg_embedding.position_embedding[:] = pos_emb_list
    print(
        "model.decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            decoder_state_dict["model.decoder.embed_positions.weight"]
            .numpy()[:max_step, :]
            .shape
        )
    )
    # assert lang in LANG2ID
    trg_tb = _gather_token_embedding(
        dec_var_name_list, decoder_state_dict, "shared", lang=lang
    )
    transformer.trg_embedding.token_embedding[:] = trg_tb.transpose().flatten().tolist()
    print(
        "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            trg_tb.transpose().shape
        )
    )

    # change encoder layer norm scale&bias position
    tmp_scale, tmp_bias = (
        transformer.src_embedding.norm_scale,
        transformer.src_embedding.norm_bias,
    )
    for i, encoder in enumerate(transformer.encoder_stack):
        print("***Fix encoder layer {} LayerNorm scale and bias***".format(i))
        new_tmp_scale, new_tmp_bias = (
            encoder.multihead_norm_scale[:],
            encoder.multihead_norm_bias[:],
        )
        encoder.multihead_norm_scale[:], encoder.multihead_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "multihead_norm_scale: {} -> {}\nmultihead_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                encoder.multihead_norm_scale[:3],
                new_tmp_bias[:3],
                encoder.multihead_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            encoder.ffn_norm_scale[:],
            encoder.ffn_norm_bias[:],
        )
        encoder.ffn_norm_scale[:], encoder.ffn_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "ffn_norm_scale: {} -> {}\nffn_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                encoder.ffn_norm_scale[:3],
                new_tmp_bias[:3],
                encoder.ffn_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]
    transformer.src_embedding.norm_scale[:], transformer.src_embedding.norm_bias[:] = (
        tmp_scale,
        tmp_bias,
    )

    # change decoder layer norm scale&bias position
    tmp_scale, tmp_bias = (
        transformer.trg_embedding.norm_scale,
        transformer.trg_embedding.norm_bias,
    )
    for i, decoder in enumerate(transformer.decoder_stack):
        print("***Fix decoder layer {} LayerNorm scale and bias***".format(i))
        new_tmp_scale, new_tmp_bias = (
            decoder.self_norm_scale[:],
            decoder.self_norm_bias[:],
        )
        decoder.self_norm_scale[:], decoder.self_norm_bias[:] = tmp_scale, tmp_bias
        print(
            "self_norm_scale: {} -> {}\nself_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.self_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.self_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            decoder.encdec_norm_scale[:],
            decoder.encdec_norm_bias[:],
        )
        decoder.encdec_norm_scale[:], decoder.encdec_norm_bias[:] = tmp_scale, tmp_bias
        print(
            "encdec_norm_scale: {} -> {}\nencdec_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.encdec_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.encdec_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            decoder.ffn_norm_scale[:],
            decoder.ffn_norm_bias[:],
        )
        decoder.ffn_norm_scale[:], decoder.ffn_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "ffn_norm_scale: {} -> {}\nffn_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.ffn_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.ffn_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]
    transformer.trg_embedding.norm_scale[:], transformer.trg_embedding.norm_bias[:] = (
        tmp_scale,
        tmp_bias,
    )

    # fill in conf

    transformer.model_conf.head_num = head_num

    transformer.model_conf.beam_size = beam_size
    transformer.model_conf.length_penalty = length_penalty

    transformer.model_conf.extra_decode_length = extra_decode_length
    transformer.model_conf.src_padding_id = 1
    transformer.model_conf.trg_start_id = 2
    transformer.model_conf.trg_end_id = 2

    transformer.model_conf.sampling_method = generation_method
    transformer.model_conf.topk = topk
    transformer.model_conf.topp = topp
    transformer.model_conf.diverse_lambda = 0
    transformer.model_conf.is_post_ln = True
    transformer.model_conf.no_scale_embedding = True
    transformer.model_conf.use_gelu = True

    print("Wrting to {0}".format(output_file))
    with tf.io.gfile.GFile(output_file, "wb") as fout:
        fout.write(transformer.SerializeToString())

    transformer = Transformer()
    with tf.io.gfile.GFile(output_file, "rb") as fin:
        transformer.ParseFromString(fin.read())
    print(transformer.model_conf)


if __name__ == "__main__":
    output_lightseq_model_name = "lightseq_bart_base.pb"
    input_huggingface_bart_model = (
        "facebook/bart-base"  ## Example: you can try "facebook/bart-large" as well
    )
    head_number = 12  # for bart-large, we have 16
    generation_method = "beam_search"  ## in order to get score, we should use `beam_search` inference method
    beam_size = 4
    max_step = 50 # max step for generation, it decides GPU memory occupancy
    extra_decode_length = 50  ## maximum_generation_length = min(src_length + extra_decode_length, max_step)
    length_penalty = 1.0
    extract_transformer_weights(
        output_lightseq_model_name,
        input_huggingface_bart_model,
        head_num=head_number,  ## layer number
        generation_method=generation_method,
        beam_size=beam_size,
        max_step=max_step,
        extra_decode_length=extra_decode_length,
        only_decoder=False,
        length_penalty=length_penalty,
    )
