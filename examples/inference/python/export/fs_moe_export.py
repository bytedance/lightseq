"""
Export Fairseq MoE models to protobuf/hdf5 format.
Pytorch implementation of MoE can be referred to https://github.com/pytorch/fairseq/tree/moe
Note that capacity is not used in LightSeq, the computational graph is dynamic.
Kernel and bias of ffn are stacked by expert_num times at the first dimension if the layer is in add_moe_list.
For example:
    the shape of ffn_first_kernel will be:
        [expert_num, hidden_size, inner_size]
    if some layer is in add_moe_list, otherwise still be:
        [hidden_size, inner_size]
"""
import os
import math
from collections import OrderedDict

import numpy as np
import torch
import tensorflow as tf
from proto.moe_pb2 import Moe
import argparse
import lightseq.inference as lsi


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
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(-2, -1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(-2, -1)",
        "ffn_second_bias": "fc2 bias",
        "gate_kernel": "moe moe gate wg weight&&expression_.transpose(0, 1)",
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
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(-2, -1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(-2, -1)",
        "ffn_second_bias": "fc2 bias",
        "gate_kernel": "moe moe gate wg weight&&expression_.transpose(0, 1)",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "layer_norm weight",
        "norm_bias": "layer_norm bias",
    }
)

trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "layer_norm weight",
        "norm_bias": "layer_norm bias",
        "shared_bias": "pred_layer bias",
    }
)


def save_proto_to_hdf5(moe, f):
    """Convert bart protobuf to hdf5 format to support larger weight."""
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
        "moe_list_encoder",
        "moe_list_decoder",
        "expert_num_encoder",
        "expert_num_decoder",
        "moe_topk_encoder",
        "moe_topk_decoder",
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
        "gate_kernel",
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
        "gate_kernel",
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

            if key not in dir(attrgetter(base_attr)(moe)):
                print(f"key {key} not found in {base_attr}, skipping")
                continue

            print(f"loading moe {proto_attr} -> {hdf5_key}")
            _data = attrgetter(proto_attr)(moe)
            if type(_data) is str:
                print(
                    f"find type str, explicitly convert string to ascii encoded array."
                )
                # explict convert to array of char (int8) to avoid issues on string reading in C
                _data = np.array([ord(c) for c in _data]).astype(np.int8)
            f.create_dataset(hdf5_key, data=_data)

    # save number of layers metadata
    f.create_dataset("model_conf/n_encoder_stack", data=len(moe.encoder_stack))
    f.create_dataset("model_conf/n_decoder_stack", data=len(moe.decoder_stack))

    # load encoder_stack
    for layer_id, layer in enumerate(moe.encoder_stack):
        for key in ENCODER_LAYER_KEYS:
            if key == "gate_kernel" and layer_id not in moe.model_conf.moe_list_encoder:
                continue
            hdf5_key = f"encoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading moe.encoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    # load decoder_stack
    for layer_id, layer in enumerate(moe.decoder_stack):
        for key in DECODER_LAYER_KEYS:
            if key == "gate_kernel" and layer_id not in moe.model_conf.moe_list_decoder:
                continue
            hdf5_key = f"decoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading moe.decoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    print(f"proto to hdf5 conversion completed.")


def check_rule(tensor_name, rule):
    if "Adam" in tensor_name or "adam" in tensor_name:
        return False
    assert isinstance(rule, str) and rule
    r_size = len(rule.split(" "))
    t = tensor_name.split(".")
    if len(t) < r_size:
        return False
    return rule == " ".join(t[-r_size:])


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
            assert len(tmp) <= 1
            target_tn.extend(tmp)
        if not target_tn and proto_name == "gate_kernel":
            continue
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
        "{0} encoder_attn k_proj weight&&{0} encoder_attn v_proj weight".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "{0} encoder_attn k_proj bias&&{0} encoder_attn v_proj bias".format(ele)
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
    """use pattern to diff source and target."""
    target_tn = []
    for tn in tensor_names:
        if (tn_pattern in tn.split(".")) and ("weight" in tn.split(".")):
            target_tn.append(tn)
            continue
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    print("token embedding shape is {}".format(target_tensor.shape))

    return target_tensor


def export_moe(
    output_file,
    model_dir,
    head_num,
    max_step,
    moe_list_encoder,
    moe_list_decoder,
    expert_num_encoder,
    expert_num_decoder,
    moe_topk_encoder,
    moe_topk_decoder,
    sampling_method="beam_search",
    extra_decode_length=50,
    beam_size=4,
    length_penalty=0.6,
    topk=1,
    topp=0.75,
    lang="en",
    bos_id=2,
    eos_id=2,
    pad_id=1,
):
    moe = Moe()
    # load var names
    reloaded = torch.load(model_dir, "cpu")
    model_dict = reloaded["model"]

    trg_emb_mapping_dict["shared_bias"] = (
        "expression_np.zeros(%d)"
        % reloaded["model"]["decoder.embed_tokens.weight"].numpy().shape[0]
    )

    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in reloaded["model"]:
        if k.startswith("encoder."):
            encoder_state_dict[k] = reloaded["model"][k]
        if k.startswith("decoder."):
            decoder_state_dict[k] = reloaded["model"][k]

    dec_var_name_list = list(decoder_state_dict.keys())
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
            moe.encoder_stack.add(),
            enc_layer_mapping_dict,
        )

    # fill each decoder layer's params
    dec_tensor_names = {}
    for name in dec_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name.split(".")[2].isdigit():
            continue
        layer_id = int(name.split(".")[2])
        dec_tensor_names.setdefault(layer_id, []).append(name)

    for layer_id in sorted(dec_tensor_names.keys()):
        fill_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            moe.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill src_embedding
    fill_layer(
        enc_var_name_list,
        encoder_state_dict,
        moe.src_embedding,
        src_emb_mapping_dict,
    )
    # encoder token embedding
    src_tb = _gather_token_embedding(
        enc_var_name_list, encoder_state_dict, "embed_tokens"
    )
    moe.src_embedding.token_embedding[:] = src_tb.flatten().tolist()
    # encoder position embedding
    pos_emb = None
    if "encoder.embed_positions.weight" in encoder_state_dict:
        pos_emb = encoder_state_dict["encoder.embed_positions.weight"].numpy()
        moe.src_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = _get_position_encoding(
            length=max_step + pad_id + 1, hidden_size=src_tb.shape[-1]
        ).numpy()
        pos_emb = pos_emb[pad_id + 1 : max_step + pad_id + 1, :]
        pos_emb_list = pos_emb.reshape([-1]).tolist()
        moe.src_embedding.position_embedding[:] = pos_emb_list

    print(
        "encoder.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    trg_emb_mapping_dict.update(encode_output_mapping_dict)
    fill_layer(
        dec_var_name_list,
        decoder_state_dict,
        moe.trg_embedding,
        trg_emb_mapping_dict,
    )
    # decoder token embedding
    trg_tb = _gather_token_embedding(
        dec_var_name_list, decoder_state_dict, "embed_tokens", lang=lang
    )
    moe.trg_embedding.token_embedding[:] = trg_tb.transpose().flatten().tolist()
    print(
        "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            trg_tb.transpose().shape
        )
    )
    # decoder position embedding
    pos_emb = None
    if "decoder.embed_positions.weight" in decoder_state_dict:
        pos_emb = decoder_state_dict["decoder.embed_positions.weight"].numpy()
        moe.trg_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = _get_position_encoding(
            length=max_step + pad_id + 1, hidden_size=trg_tb.shape[-1]
        ).numpy()
        pos_emb = pos_emb[pad_id + 1 : max_step + pad_id + 1, :]
        pos_emb_list = pos_emb.reshape([-1]).tolist()
        moe.trg_embedding.position_embedding[:] = pos_emb_list

    print(
        "decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill in conf
    moe.model_conf.head_num = head_num

    moe.model_conf.beam_size = beam_size
    moe.model_conf.length_penalty = length_penalty

    moe.model_conf.extra_decode_length = extra_decode_length
    moe.model_conf.src_padding_id = pad_id
    moe.model_conf.trg_start_id = bos_id
    moe.model_conf.trg_end_id = eos_id

    moe.model_conf.sampling_method = sampling_method
    moe.model_conf.topk = topk
    moe.model_conf.topp = topp
    moe.model_conf.diverse_lambda = 0
    moe.model_conf.is_post_ln = False
    moe.model_conf.no_scale_embedding = False
    moe.model_conf.use_gelu = False

    moe.model_conf.moe_list_encoder[:] = moe_list_encoder
    moe.model_conf.moe_list_decoder[:] = moe_list_decoder
    moe.model_conf.expert_num_encoder = expert_num_encoder
    moe.model_conf.expert_num_decoder = expert_num_decoder
    moe.model_conf.moe_topk_encoder = moe_topk_encoder
    moe.model_conf.moe_topk_decoder = moe_topk_decoder

    _write(moe, output_file)


def _write(moe, path):
    print("Wrting to {0}".format(path))

    try:
        with tf.io.gfile.GFile(path, "wb") as fout:
            fout.write(moe.SerializeToString())
    except Exception:
        print("Saving PB fails. Save HDF5 instead!")
        if os.path.exists(path):
            os.remove(path)
        path = path.replace("pb", "hdf5")
        import h5py

        f = h5py.File(path, "w")
        save_proto_to_hdf5(moe, f)
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="create data for post-model training", usage=""
    )
    parser.add_argument(
        "--input", type=str, default="checkpoint.pt", help="input fairseq checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="moe.pb", help="output lightseq model file"
    )
    parser.add_argument("--beam_size", type=int, default=4, help="beam size")
    parser.add_argument("--max-step", type=int, default=1024, help="max step to decode")
    parser.add_argument("--head-num", type=int, default=16, help="head num")
    parser.add_argument("--bos_id", type=int, default=2, help="bos id")
    parser.add_argument("--eos_id", type=int, default=2, help="eos id")
    parser.add_argument("--pad_id", type=int, default=1, help="pad id")
    # newly added for MoE
    parser.add_argument(
        "--moe_list_encoder",
        type=list,
        default=[0, 1, 2, 3, 4, 5],
        help="list of encoder layers that enable MoE module",
    )
    parser.add_argument(
        "--moe_list_decoder",
        type=list,
        default=[],
        help="list of decoder layers that enable MoE module",
    )
    parser.add_argument(
        "--expert_num_encoder", type=int, default=8, help="number of experts in encoder"
    )
    parser.add_argument(
        "--expert_num_decoder", type=int, default=8, help="number of experts in decoder"
    )
    parser.add_argument(
        "--moe_topk_encoder",
        type=int,
        default=2,
        help="topk of MoE gate in encoder, support k=1 or 2",
    )
    parser.add_argument(
        "--moe_topk_decoder",
        type=int,
        default=2,
        help="topk of MoE gate in decoder, support k=1 or 2",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # export
    export_moe(
        args.output,
        args.input,
        args.head_num,
        beam_size=args.beam_size,
        max_step=args.max_step,
        lang="en",
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        pad_id=args.pad_id,
        moe_list_encoder=args.moe_list_encoder,
        moe_list_decoder=args.moe_list_decoder,
        expert_num_encoder=args.expert_num_encoder,
        expert_num_decoder=args.expert_num_decoder,
        moe_topk_encoder=args.moe_topk_encoder,
        moe_topk_decoder=args.moe_topk_decoder,
    )
    # test
    model_path = None
    if os.path.exists(args.output):
        model_path = args.output
    elif os.path.exists(args.output.replace("pb", "hdf5")):
        model_path = args.output.replace("pb", "hdf5")
    if model_path:
        ls_model = lsi.Moe(model_path, 8)
        # Note that pad_id (1) should be on the right side
        # wmt14 en2de translation
        src_tokens = np.array([[10827, 27, 2081, 2], [1478, 6, 2, 1]])
        trg_out = ls_model.infer(src_tokens)
        print(trg_out)
