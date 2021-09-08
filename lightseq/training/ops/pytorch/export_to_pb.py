from collections import OrderedDict
import numpy as np


def _gen_enc_offset(hidden_size, intermediate_size):
    hs, ims = hidden_size, intermediate_size
    sizes = [
        hs * hs * 3,  # attn_qkvw
        hs * 3,  # attn_qkvb
        hs * hs,  # attn_ow
        hs,  # attn_ob
        hs,  # attn_nw
        hs,  # attn_nb
        hs * ims,  # inter_w
        ims,  # inter_b
        hs * ims,  # output_w
        hs,  # output_b
        hs,  # ffn_nw
        hs,  # ffn_nb
    ]
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets

def _gen_dec_offset(hidden_size, intermediate_size, nlayer):
    hs, ims = hidden_size, intermediate_size
    sizes = [
        hs * hs * 3,  # attn_qkvw
        hs * 3,  # attn_qkvb
        hs * hs,  # attn_ow
        hs,  # attn_ob
        hs,  # attn_nw
        hs,  # attn_nb
        hs * hs,  # encdec_attn_qw
        hs,  # encdec_attn_qb
        hs * hs,  # encdec_attn_ow
        hs,  # encdec_attn_ob
        hs,  # encdec_attn_nw
        hs,  # encdec_attn_nb
        hs * ims,  # inter_w
        ims,  # inter_b
        hs * ims,  # output_w
        hs,  # output_b
        hs,  # ffn_nw
        hs,  # ffn_nb
        hs * hs * 2 * nlayer,  # encdec_attn_kvw
        hs * 2 * nlayer,  # encdec_attn_kvb
    ]
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets

def _check_rule(tensor_name, rule):
    if "Adam" in tensor_name or "adam" in tensor_name:
        return False
    assert isinstance(rule, str) and rule
    r_size = len(rule.split('.'))
    t = tensor_name.split('.')
    if len(t) < r_size:
        return False
    return rule == '.'.join(t[-r_size:])

def _fill_layer(tensor_names, state_dict, layer, mapping_dict):
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
                if _check_rule(tn, cr):
                    tmp.append(tn)
            assert len(tmp) == 1
            target_tn.extend(tmp)
        target_tensor = [state_dict[name] for name in target_tn]
        tt = {}
        if target_tensor:
            exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)
        else:
            if not isinstance(expression, list):
                expression = [expression]
            exec("tt['save'] = [%s]" % ",".join(expression))

        target_tensor = np.concatenate(tt["save"], axis=-1)
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)

def _fill_encdec_weight(transformer, state_dict, mapping_dict, is_encoder, enc_out_mapping_dict=None):
    var_name_list = list(state_dict.keys())

    tensor_names = {}
    for name in var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name.split(".")[2].isdigit():
            continue
        layer_id = int(name.split(".")[2])
        tensor_names.setdefault(layer_id, []).append(name)
    
    if is_encoder:
        layer = transformer.encoder_stack.add()
    else:
        layer = transformer.decoder_stack.add()

    for layer_id in sorted(tensor_names.keys()):
        _fill_layer(
            tensor_names[layer_id],
            state_dict,
            layer,
            mapping_dict,
        )

    if not is_encoder:
        _fill_layer(
            tensor_names[0],
            state_dict,
            transformer.trg_embedding,
            enc_out_mapping_dict,
        )

def export_encoder(transformer, state_dict, hidden_size, intermediate_size):
    hs, ims = hidden_size, intermediate_size
    offsets = _gen_enc_offset(hs, ims)
    mapping_dict = OrderedDict(
        {
            "multihead_project_kernel_qkv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[0], offsets[1], 3*hs, hs),
            "multihead_project_bias_qkv": "para&&expression_[{0}:{1}]".format(offsets[1], offsets[2]),
            "multihead_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[2], offsets[3], hs, hs),
            "multihead_project_bias_output": "para&&expression_[{0}:{1}]".format(offsets[3], offsets[4]),
            "multihead_norm_scale": "para&&expression_[{0}:{1}]".format(offsets[4], offsets[5]),
            "multihead_norm_bias": "para&&expression_[{0}:{1}]".format(offsets[5], offsets[6]),
            "ffn_first_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[6], offsets[7], ims, hs),
            "ffn_first_bias": "para&&expression_[{0}:{1}]".format(offsets[7], offsets[8]),
            "ffn_second_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[8], offsets[9], hs, ims),
            "ffn_second_bias": "para&&expression_[{0}:{1}]".format(offsets[9], offsets[10]),
            "ffn_norm_scale": "para&&expression_[{0}:{1}]".format(offsets[10], offsets[11]),
            "ffn_norm_bias": "para&&expression_[{0}:{1}]".format(offsets[11], offsets[12]),
        }
    )
    _fill_encdec_weight(transformer, state_dict, mapping_dict, True)

def export_decoder(transformer, state_dict, hidden_size, intermediate_size, nlayer):
    hs, ims = hidden_size, intermediate_size
    offsets = _gen_dec_offset(hs, ims, nlayer)
    mapping_dict = OrderedDict(
        {
            "self_project_kernel_qkv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[0], offsets[1], 3*hs, hs),
            "self_project_bias_qkv": "para&&expression_[{0}:{1}]".format(offsets[1], offsets[2]),
            "self_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[2], offsets[3], hs, hs),
            "self_project_bias_output": "para&&expression_[{0}:{1}]".format(offsets[3], offsets[4]),
            "self_norm_scale": "para&&expression_[{0}:{1}]".format(offsets[4], offsets[5]),
            "self_norm_bias": "para&&expression_[{0}:{1}]".format(offsets[5], offsets[6]),
            "encdec_project_kernel_q": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[6], offsets[7], hs, hs),
            "encdec_project_bias_q": "para&&expression_[{0}:{1}]".format(offsets[7], offsets[8]),
            "encdec_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[8], offsets[9], hs, hs),
            "encdec_project_bias_output": "para&&expression_[{0}:{1}]".format(offsets[9], offsets[10]),
            "encdec_norm_scale": "para&&expression_[{0}:{1}]".format(offsets[10], offsets[11]),
            "encdec_norm_bias": "para&&expression_[{0}:{1}]".format(offsets[11], offsets[12]),
            "ffn_first_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[12], offsets[13], ims, hs),
            "ffn_first_bias": "para&&expression_[{0}:{1}]".format(offsets[13], offsets[14]),
            "ffn_second_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[14], offsets[15], hs, ims),
            "ffn_second_bias": "para&&expression_[{0}:{1}]".format(offsets[15], offsets[16]),
            "ffn_norm_scale": "para&&expression_[{0}:{1}]".format(offsets[16], offsets[17]),
            "ffn_norm_bias": "para&&expression_[{0}:{1}]".format(offsets[17], offsets[18]),
        }
    )
    enc_out_mapping_dict = OrderedDict(
        {
            "encode_output_project_kernel_kv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(offsets[18], offsets[19], 2*nlayer*hs, hs),
            "encode_output_project_bias_kv": "para&&expression_[{0}:{1}]".format(offsets[19], offsets[20]),
        }
    )
    _fill_encdec_weight(transformer, state_dict, mapping_dict, False, enc_out_mapping_dict)

def export_config(transformer,
                  nhead,
                  pad_id,
                  start_id,
                  end_id,
                  is_post_ln=False,
                  no_scale_embedding=False,
                  use_gelu=False,
                  beam_size=4,
                  length_penalty=0.6,
                  extra_decode_length=50,
                  generation_method='beam_search',
                  topk=1,
                  topp=0.75,
                  diverse_lambda=0):
    transformer.model_conf.head_num = nhead
    transformer.model_conf.src_padding_id = pad_id
    transformer.model_conf.trg_start_id = start_id
    transformer.model_conf.trg_end_id = end_id
    transformer.model_conf.is_post_ln = is_post_ln
    transformer.model_conf.no_scale_embedding = no_scale_embedding
    transformer.model_conf.use_gelu = use_gelu

    transformer.model_conf.beam_size = beam_size
    transformer.model_conf.length_penalty = length_penalty
    transformer.model_conf.extra_decode_length = extra_decode_length
    transformer.model_conf.sampling_method = generation_method
    transformer.model_conf.topk = topk
    transformer.model_conf.topp = topp
    transformer.model_conf.diverse_lambda = diverse_lambda
    