from collections import OrderedDict
import numpy as np
from lightseq.training.ops.pytorch.util import get_pos_embedding
from lightseq.training import LSTransformerEncoderLayer, LSTransformerDecoderLayer
from lightseq.training.ops.pytorch.export import apply_rule


global_quant_range = 127
global_act_clip_max = 16.0


def quantize(tensor, quant_range, clip_max):
    return np.floor(
        (
            np.clip(tensor * quant_range / clip_max, -quant_range, quant_range)
            + quant_range
        )
        + 0.5
    ).astype(np.ubyte)


def get_kth_value(tensor, k=None):
    if k is None:
        k = max(int(tensor.size / 50000.0), 1)
    return max(np.partition(tensor.flatten(), -int(k))[-int(k)], 0.0)


def gather_quant_token_embedding(tensor_names, state_dict, tn_pattern, clip_max=None):
    target_tn = []
    for tn in tensor_names:
        if tn_pattern in tn.split("."):
            target_tn.append(tn)
            continue
    target_tensor = [state_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    target_tensor = target_tensor.astype(np.float16).astype(np.float32)
    if clip_max is None:
        clip_max = get_kth_value(target_tensor)
    target_tensor = quantize(target_tensor, global_quant_range, clip_max)
    return target_tensor, clip_max, target_tn


def fill_quant_pb_layer(
    tensor_names,
    state_dict,
    layer,
    mapping_dict,
    act_clip_max,
    nlayer=None,
):
    for proto_name, ckpt_rule in mapping_dict.items():
        if "clip_max" in proto_name and "kernel" not in proto_name:
            exec("layer.%s=act_clip_max" % proto_name)
            print("%s convert finished!" % proto_name)
            continue
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        if "kernel" in proto_name:
            weight_clip_max = get_kth_value(target_tensor)
            target_tensor = quantize(target_tensor, global_quant_range, weight_clip_max)
            exec("layer.%s=bytes(target_tensor.flatten().tolist())" % proto_name)
            if proto_name == "encode_output_project_kernel_kv":
                assert nlayer is not None
                clip_maxs = [weight_clip_max] * int(nlayer)
                exec("layer.%s_clip_max[:]=clip_maxs" % proto_name)
            else:
                exec("layer.%s_clip_max=weight_clip_max" % proto_name)
            print("%s_clip_max convert finished!" % proto_name)
        else:
            exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def fill_encdec_weight(
    file,
    state_dict,
    mapping_dict,
    act_clip_max,
    is_encoder,
    save_pb,
    enc_out_mapping_dict=None,
    nlayer=None,
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
            layer = file.encoder_stack.add()
        else:
            layer = file.decoder_stack.add()
        fill_quant_pb_layer(
            tensor_names[layer_id],
            state_dict,
            layer,
            mapping_dict,
            act_clip_max,
        )

    if not is_encoder:
        fill_quant_pb_layer(
            tensor_names[0],
            state_dict,
            file.trg_embedding,
            enc_out_mapping_dict,
            act_clip_max,
            nlayer,
        )


def export_ls_embedding_ptq(
    file,
    state_dict,
    max_length,
    is_encoder,
    save_pb=True,
):
    var_name_list = list(state_dict.keys())
    emb, clip_max, target_tn = gather_quant_token_embedding(
        var_name_list, state_dict, "embeddings"
    )
    if is_encoder:
        emb_list = emb.flatten().tolist()
        file.src_embedding.token_embedding = bytes(emb_list)
        file.src_embedding.emb_clip_max = clip_max
    else:
        emb_list = emb.transpose().flatten().tolist()
        file.trg_embedding.token_embedding = bytes(emb_list)
        file.trg_embedding.emb_clip_max = clip_max
    print(
        "%s -> %s_embedding.token_embedding, convert finished!"
        % (target_tn, "src" if is_encoder else "trg")
    )
    print("%s emb_clip_max convert finished!" % target_tn)

    pos_emb = get_pos_embedding(max_length, emb.shape[-1])
    pos_emb_list = pos_emb.flatten().tolist()
    if is_encoder:
        file.src_embedding.position_embedding[:] = pos_emb_list
    else:
        file.trg_embedding.position_embedding[:] = pos_emb_list
    target_tn = [tn.replace("embeddings", "pos_embeddings") for tn in target_tn]
    print(
        "%s -> %s_embedding.position_embedding, convert finished!"
        % (target_tn, "src" if is_encoder else "trg")
    )


def export_ls_encoder_ptq(
    file,
    state_dict,
    hidden_size,
    intermediate_size,
    act_clip_max=global_act_clip_max,
    save_pb=True,
):
    hs, ims = hidden_size, intermediate_size
    offsets = LSTransformerEncoderLayer.gen_offset(hs, ims)
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
            "multihead_ln_clip_max": "None",
            "multihead_project_output_clip_max": "None",
            "ffn_ln_clip_max": "None",
            "ffn_first_act_clip_max": "None",
            "multihead_qkv_dense_clip_max": "None",
            "multihead_output_dense_clip_max": "None",
            "ffn_first_output_clip_max": "None",
        }
    )
    fill_encdec_weight(file, state_dict, mapping_dict, act_clip_max, True, save_pb)


def export_ls_decoder_ptq(
    file,
    state_dict,
    hidden_size,
    intermediate_size,
    nlayer,
    act_clip_max=global_act_clip_max,
    save_pb=True,
):
    hs, ims = hidden_size, intermediate_size
    offsets = LSTransformerDecoderLayer.gen_offset(hs, ims, nlayer)
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
            "self_ln_clip_max": "None",
            "self_project_output_clip_max": "None",
            "encdec_ln_clip_max": "None",
            "encdec_project_output_clip_max": "None",
            "ffn_ln_clip_max": "None",
            "ffn_first_act_clip_max": "None",
            "self_qkv_dense_clip_max": "None",
            "self_output_dense_clip_max": "None",
            "encdec_q_dense_clip_max": "None",
            "encdec_output_dense_clip_max": "None",
            "ffn_first_output_clip_max": "None",
            "self_qkv_bias_out_clip_max": "None",
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
            "output_ln_clip_max": "None",
            "logits_clip_max": "None",
        }
    )
    fill_encdec_weight(
        file,
        state_dict,
        mapping_dict,
        act_clip_max,
        False,
        save_pb,
        enc_out_mapping_dict,
        nlayer,
    )
