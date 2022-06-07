from collections import OrderedDict
import numpy as np
from lightseq.training.ops.pytorch.util import get_pos_embedding
from lightseq.training import LSTransformerEncoderLayer, LSTransformerDecoderLayer


def gather_token_embedding(tensor_names, state_dict, tn_pattern, scale=True):
    target_tn = []
    for tn in tensor_names:
        if tn_pattern in tn.split("."):
            target_tn.append(tn)
            continue
    target_tensor = [state_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    if scale:
        target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    return target_tensor, target_tn


def apply_rule(proto_name, ckpt_rule, tensor_names, state_dict):
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

    expression = [ele for ele in ckpt_rule.split("&&") if ele.startswith("expression_")]

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

    try:
        target_tensor = np.concatenate(tt["save"], axis=-1)
    except:
        target_tensor = tt["save"]
    print(
        "%s -> %s, convert finished!"
        % (target_tn if target_tn else "created", proto_name)
    )
    return target_tensor


def fill_pb_layer(tensor_names, state_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def fill_hdf5_layer(
    tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix, mapping_dict
):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        hdf5_file.create_dataset(
            hdf5_dataset_prefix + proto_name, data=target_tensor.flatten().tolist()
        )


def fill_encdec_weight(
    file, state_dict, mapping_dict, is_encoder, save_pb, enc_out_mapping_dict=None
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
        if save_pb:
            if is_encoder:
                layer = file.encoder_stack.add()
            else:
                layer = file.decoder_stack.add()
            fill_pb_layer(
                tensor_names[layer_id],
                state_dict,
                layer,
                mapping_dict,
            )
        else:
            if is_encoder:
                dataset_prefix = f"encoder_stack/{layer_id}/"
            else:
                dataset_prefix = f"decoder_stack/{layer_id}/"
            fill_hdf5_layer(
                tensor_names[layer_id],
                state_dict,
                file,
                dataset_prefix,
                mapping_dict,
            )

    if not is_encoder:
        if save_pb:
            fill_pb_layer(
                tensor_names[0],
                state_dict,
                file.trg_embedding,
                enc_out_mapping_dict,
            )
        else:
            fill_hdf5_layer(
                tensor_names[0],
                state_dict,
                file,
                f"trg_embedding/",
                enc_out_mapping_dict,
            )


def export_ls_embedding(file, state_dict, max_length, is_encoder, save_pb=True):
    var_name_list = list(state_dict.keys())
    emb, target_tn = gather_token_embedding(var_name_list, state_dict, "embeddings")
    if is_encoder:
        emb_list = emb.flatten().tolist()
        if save_pb:
            file.src_embedding.token_embedding[:] = emb_list
        else:
            file.create_dataset(
                "src_embedding/token_embedding", data=emb_list, dtype="f4"
            )
    else:
        emb_list = emb.transpose().flatten().tolist()
        if save_pb:
            file.trg_embedding.token_embedding[:] = emb_list
        else:
            file.create_dataset(
                "trg_embedding/token_embedding", data=emb_list, dtype="f4"
            )
    print(
        "%s -> %s_embedding.token_embedding, convert finished!"
        % (target_tn, "src" if is_encoder else "trg")
    )

    pos_emb = get_pos_embedding(max_length, emb.shape[-1])
    pos_emb_list = pos_emb.flatten().tolist()
    if is_encoder:
        if save_pb:
            file.src_embedding.position_embedding[:] = pos_emb_list
        else:
            file.create_dataset(
                "src_embedding/position_embedding", data=pos_emb_list, dtype="f4"
            )
    else:
        if save_pb:
            file.trg_embedding.position_embedding[:] = pos_emb_list
        else:
            file.create_dataset(
                "trg_embedding/position_embedding", data=pos_emb_list, dtype="f4"
            )
    target_tn = [tn.replace("embeddings", "pos_embeddings") for tn in target_tn]
    print(
        "%s -> %s_embedding.position_embedding, convert finished!"
        % (target_tn, "src" if is_encoder else "trg")
    )


def export_ls_encoder(file, state_dict, hidden_size, intermediate_size, save_pb=True):
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
        }
    )
    fill_encdec_weight(file, state_dict, mapping_dict, True, save_pb)


def export_ls_decoder(
    file, state_dict, hidden_size, intermediate_size, nlayer, save_pb=True
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
    fill_encdec_weight(
        file, state_dict, mapping_dict, False, save_pb, enc_out_mapping_dict
    )


def export_ls_config(
    file,
    head_num,
    src_padding_id,
    trg_start_id,
    trg_end_id,
    n_encoder_stack,
    n_decoder_stack,
    is_post_ln=False,
    no_scale_embedding=False,
    use_gelu=False,
    beam_size=4,
    length_penalty=0.6,
    extra_decode_length=50,
    sampling_method="beam_search",
    topk=1,
    topp=0.75,
    diverse_lambda=0,
    save_pb=True,
):
    args = locals()
    args.pop("file")
    args.pop("save_pb")
    if save_pb:
        args.pop("n_encoder_stack")
        args.pop("n_decoder_stack")
        for v in list(args.keys()):
            exec("file.model_conf.{0} = {0}".format(v))
    else:
        sampling_method = np.array([ord(c) for c in sampling_method]).astype(np.int8)
        for v in list(args.keys()):
            exec("file.create_dataset('model_conf/{0}', data={0})".format(v))


def export_pb2hdf5(transformer, f):
    """Convert Transformer protobuf to hdf5 format to support larger weight."""
    MODEL_CONF_KEYS = [
        # model_conf
        "head_num",
        "beam_size",
        "extra_decode_length",
        "length_penalty",
        "src_padding_id",
        "trg_start_id",
        "diverse_lambda",
        "sampling_method",
        "topp",
        "topk",
        "trg_end_id",
        "is_post_ln",
        "no_scale_embedding",
        "use_gelu",
        "multilg_type",
    ]

    EMBEDDING_KEYS = [
        # src_embedding
        # trg_embedding
        "token_embedding",
        "position_embedding",
        "norm_scale",
        "norm_bias",
        "encode_output_project_kernel_kv",
        "encode_output_project_bias_kv",
        "shared_bias",
        "lang_emb",
        "trg_vocab_mask",
    ]

    ENCODER_LAYER_KEYS = [
        # encoder_stack/{i}
        "multihead_norm_scale",
        "multihead_norm_bias",
        "multihead_project_kernel_qkv",
        "multihead_project_bias_qkv",
        "multihead_project_kernel_output",
        "multihead_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]

    DECODER_LAYER_KEYS = [
        # decoder_stack/{i}
        "self_norm_scale",
        "self_norm_bias",
        "self_project_kernel_qkv",
        "self_project_bias_qkv",
        "self_project_kernel_output",
        "self_project_bias_output",
        "encdec_norm_scale",
        "encdec_norm_bias",
        "encdec_project_kernel_q",
        "encdec_project_bias_q",
        "encdec_project_kernel_output",
        "encdec_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]
    base_attr_to_keys = {
        "src_embedding": EMBEDDING_KEYS,
        "trg_embedding": EMBEDDING_KEYS,
        "model_conf": MODEL_CONF_KEYS,
    }

    from operator import attrgetter

    print(f"start converting protobuf to hdf5 format.")
    # load src_embedding, trg_embedding, model_conf
    for base_attr, keys in base_attr_to_keys.items():
        for key in keys:
            hdf5_key = f"{base_attr}/{key}"
            proto_attr = f"{base_attr}.{key}"

            if key not in dir(attrgetter(base_attr)(transformer)):
                print(f"key {key} not found in {base_attr}, skipping")
                continue

            print(f"loading transformer {proto_attr} -> {hdf5_key}")
            _data = attrgetter(proto_attr)(transformer)
            if type(_data) is str:
                print(
                    f"find type str, explicitly convert string to ascii encoded array."
                )
                # explict convert to array of char (int8) to avoid issues on string reading in C
                _data = np.array([ord(c) for c in _data]).astype(np.int8)
            f.create_dataset(hdf5_key, data=_data)

    # save number of layers metadata
    f.create_dataset("model_conf/n_encoder_stack", data=len(transformer.encoder_stack))
    f.create_dataset("model_conf/n_decoder_stack", data=len(transformer.decoder_stack))

    # load encoder_stack
    for layer_id, layer in enumerate(transformer.encoder_stack):
        for key in ENCODER_LAYER_KEYS:
            hdf5_key = f"encoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading transformer.encoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    # load decoder_stack
    for layer_id, layer in enumerate(transformer.decoder_stack):
        for key in DECODER_LAYER_KEYS:
            hdf5_key = f"decoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading transformer.decoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    print(f"proto to hdf5 conversion completed.")
