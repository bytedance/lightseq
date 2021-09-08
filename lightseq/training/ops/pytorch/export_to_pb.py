from collections import OrderedDict
import math
import logging
import numpy as np
import torch

logging.getLogger().setLevel(logging.INFO)


def _calc_offset(sizes):
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets


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
    offsets = _calc_offset(sizes)
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
    offsets = _calc_offset(sizes)
    return offsets


def _fill_layer(tensor_names, state_dict, layer, mapping_dict):
    def _check_rule(tensor_name, rule):
        if "Adam" in tensor_name or "adam" in tensor_name:
            return False
        assert isinstance(rule, str) and rule
        r_size = len(rule.split("."))
        t = tensor_name.split(".")
        if len(t) < r_size:
            return False
        return rule == ".".join(t[-r_size:])

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
        logging.info(
            "%s -> %s, convert finished."
            % (target_tn if target_tn else "created", proto_name)
        )
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def _fill_encdec_weight(
    transformer, state_dict, mapping_dict, is_encoder, enc_out_mapping_dict=None
):
    var_name_list = list(state_dict.keys())

    tensor_names = {}
    for name in var_name_list:
        name_split = name.split(".")
        # assert the layer name like `xxx.0.xxx.para`
        if name_split[-1] != "para":
            continue
        for s in name_split[::-1]:
            if s.isdigit():
                tensor_names.setdefault(int(s), []).append(name)
                break
    assert len(tensor_names) > 0

    for layer_id in sorted(tensor_names.keys()):
        if is_encoder:
            layer = transformer.encoder_stack.add()
        else:
            layer = transformer.decoder_stack.add()
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


def _gather_token_embedding(tensor_names, state_dict, tn_pattern):
    target_tn = []
    for tn in tensor_names:
        if tn_pattern == tn.split(".")[-1]:
            target_tn.append(tn)
            continue
    target_tensor = [state_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    return target_tensor, target_tn


def _get_pos_embedding(max_length, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
    return emb


def export_ls_embedding(transformer, state_dict, max_length, is_encoder):
    var_name_list = list(state_dict.keys())
    emb, target_tn = _gather_token_embedding(var_name_list, state_dict, "embeddings")
    if is_encoder:
        transformer.src_embedding.token_embedding[:] = emb.flatten().tolist()
    else:
        transformer.trg_embedding.token_embedding[:] = (
            emb.transpose().flatten().tolist()
        )
    logging.info(
        "%s -> %s_embedding.token_embedding, convert finished."
        % (target_tn, "src" if is_encoder else "trg")
    )

    pos_emb = _get_pos_embedding(max_length, emb.shape[-1])
    if is_encoder:
        transformer.src_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        transformer.trg_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    target_tn = [tn.replace("embeddings", "pos_embeddings") for tn in target_tn]
    logging.info(
        "%s -> %s_embedding.position_embedding, convert finished."
        % (target_tn, "src" if is_encoder else "trg")
    )


def export_ls_encoder(transformer, state_dict, hidden_size, intermediate_size):
    hs, ims = hidden_size, intermediate_size
    offsets = _gen_enc_offset(hs, ims)
    mapping_dict = OrderedDict(
        {
            "multihead_project_kernel_qkv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[0], offsets[1], 3 * hs, hs
            ),
            "multihead_project_bias_qkv": "para&&expression_[{0}:{1}]".format(
                offsets[1], offsets[2]
            ),
            "multihead_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[2], offsets[3], hs, hs
            ),
            "multihead_project_bias_output": "para&&expression_[{0}:{1}]".format(
                offsets[3], offsets[4]
            ),
            "multihead_norm_scale": "para&&expression_[{0}:{1}]".format(
                offsets[4], offsets[5]
            ),
            "multihead_norm_bias": "para&&expression_[{0}:{1}]".format(
                offsets[5], offsets[6]
            ),
            "ffn_first_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[6], offsets[7], ims, hs
            ),
            "ffn_first_bias": "para&&expression_[{0}:{1}]".format(
                offsets[7], offsets[8]
            ),
            "ffn_second_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[8], offsets[9], hs, ims
            ),
            "ffn_second_bias": "para&&expression_[{0}:{1}]".format(
                offsets[9], offsets[10]
            ),
            "ffn_norm_scale": "para&&expression_[{0}:{1}]".format(
                offsets[10], offsets[11]
            ),
            "ffn_norm_bias": "para&&expression_[{0}:{1}]".format(
                offsets[11], offsets[12]
            ),
        }
    )
    _fill_encdec_weight(transformer, state_dict, mapping_dict, True)


def export_ls_decoder(transformer, state_dict, hidden_size, intermediate_size, nlayer):
    hs, ims = hidden_size, intermediate_size
    offsets = _gen_dec_offset(hs, ims, nlayer)
    mapping_dict = OrderedDict(
        {
            "self_project_kernel_qkv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[0], offsets[1], 3 * hs, hs
            ),
            "self_project_bias_qkv": "para&&expression_[{0}:{1}]".format(
                offsets[1], offsets[2]
            ),
            "self_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[2], offsets[3], hs, hs
            ),
            "self_project_bias_output": "para&&expression_[{0}:{1}]".format(
                offsets[3], offsets[4]
            ),
            "self_norm_scale": "para&&expression_[{0}:{1}]".format(
                offsets[4], offsets[5]
            ),
            "self_norm_bias": "para&&expression_[{0}:{1}]".format(
                offsets[5], offsets[6]
            ),
            "encdec_project_kernel_q": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[6], offsets[7], hs, hs
            ),
            "encdec_project_bias_q": "para&&expression_[{0}:{1}]".format(
                offsets[7], offsets[8]
            ),
            "encdec_project_kernel_output": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[8], offsets[9], hs, hs
            ),
            "encdec_project_bias_output": "para&&expression_[{0}:{1}]".format(
                offsets[9], offsets[10]
            ),
            "encdec_norm_scale": "para&&expression_[{0}:{1}]".format(
                offsets[10], offsets[11]
            ),
            "encdec_norm_bias": "para&&expression_[{0}:{1}]".format(
                offsets[11], offsets[12]
            ),
            "ffn_first_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[12], offsets[13], ims, hs
            ),
            "ffn_first_bias": "para&&expression_[{0}:{1}]".format(
                offsets[13], offsets[14]
            ),
            "ffn_second_kernel": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[14], offsets[15], hs, ims
            ),
            "ffn_second_bias": "para&&expression_[{0}:{1}]".format(
                offsets[15], offsets[16]
            ),
            "ffn_norm_scale": "para&&expression_[{0}:{1}]".format(
                offsets[16], offsets[17]
            ),
            "ffn_norm_bias": "para&&expression_[{0}:{1}]".format(
                offsets[17], offsets[18]
            ),
        }
    )
    enc_out_mapping_dict = OrderedDict(
        {
            "encode_output_project_kernel_kv": "para&&expression_[{0}:{1}].reshape({2}, {3}).transpose(0, 1)".format(
                offsets[18], offsets[19], 2 * nlayer * hs, hs
            ),
            "encode_output_project_bias_kv": "para&&expression_[{0}:{1}]".format(
                offsets[19], offsets[20]
            ),
        }
    )
    _fill_encdec_weight(
        transformer, state_dict, mapping_dict, False, enc_out_mapping_dict
    )


def export_ls_config(
    transformer,
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
    generation_method="beam_search",
    topk=1,
    topp=0.75,
    diverse_lambda=0,
):
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
