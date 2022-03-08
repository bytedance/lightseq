"""
Export Fairseq Transformer models to protobuf/hdf5 format.
"""
import os
import torch
import argparse
import numpy as np
from collections import OrderedDict

import lightseq.inference as lsi
from proto.transformer_pb2 import Transformer
from lightseq.training.ops.pytorch.export import (
    gather_token_embedding,
    fill_pb_layer,
    export_ls_config,
)
from lightseq.training.ops.pytorch.util import get_pos_embedding


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


def save_proto_to_hdf5(transformer, f):
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


def export_native_fs_transformer(
    output_file,
    model_dir,
    args,
):
    transformer = Transformer()
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
        fill_pb_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            transformer.encoder_stack.add(),
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
        fill_pb_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill src_embedding
    fill_pb_layer(
        enc_var_name_list,
        encoder_state_dict,
        transformer.src_embedding,
        src_emb_mapping_dict,
    )
    # encoder token embedding
    src_tb, _ = gather_token_embedding(
        enc_var_name_list, encoder_state_dict, "embed_tokens"
    )
    transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()
    # encoder position embedding
    pos_emb = None
    if "encoder.embed_positions.weight" in encoder_state_dict:
        pos_emb = encoder_state_dict["encoder.embed_positions.weight"].numpy()
        transformer.src_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = get_pos_embedding(
            args.max_step + args.pad_id + 1, src_tb.shape[-1]
        ).numpy()
        pos_emb = pos_emb[args.pad_id + 1 : args.max_step + args.pad_id + 1, :]
        pos_emb_list = pos_emb.reshape([-1]).tolist()
        transformer.src_embedding.position_embedding[:] = pos_emb_list

    print(
        "encoder.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    trg_emb_mapping_dict.update(encode_output_mapping_dict)
    fill_pb_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        trg_emb_mapping_dict,
    )
    # decoder token embedding
    trg_tb, _ = gather_token_embedding(
        dec_var_name_list, decoder_state_dict, "embed_tokens"
    )
    transformer.trg_embedding.token_embedding[:] = trg_tb.transpose().flatten().tolist()
    print(
        "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            trg_tb.transpose().shape
        )
    )
    # decoder position embedding
    pos_emb = None
    if "decoder.embed_positions.weight" in decoder_state_dict:
        pos_emb = decoder_state_dict["decoder.embed_positions.weight"].numpy()
        transformer.trg_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = get_pos_embedding(
            args.max_step + args.pad_id + 1, trg_tb.shape[-1]
        ).numpy()
        pos_emb = pos_emb[args.pad_id + 1 : args.max_step + args.pad_id + 1, :]
        pos_emb_list = pos_emb.reshape([-1]).tolist()
        transformer.trg_embedding.position_embedding[:] = pos_emb_list

    print(
        "decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill in conf
    export_ls_config(
        transformer,
        args.head_num,
        args.pad_id,
        args.bos_id,
        args.eos_id,
        0,
        0,
        beam_size=args.beam_size,
        save_pb=True,
    )

    _write(transformer, output_file)


def _write(transformer, path):
    print("Wrting to {0}".format(path))

    try:
        with open(path, "wb") as fout:
            fout.write(transformer.SerializeToString())
    except Exception:
        print("Saving PB fails. Save HDF5 instead!")
        if os.path.exists(path):
            os.remove(path)
        path = path.replace("pb", "hdf5")
        import h5py

        f = h5py.File(path, "w")
        save_proto_to_hdf5(transformer, f)
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="create data for post-model training", usage=""
    )
    parser.add_argument(
        "--input", type=str, default="checkpoint.pt", help="input fairseq checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transformer.pb",
        help="output lightseq model file",
    )
    parser.add_argument("--beam_size", type=int, default=4, help="beam size")
    parser.add_argument("--max_step", type=int, default=1024, help="max step to decode")
    parser.add_argument("--head_num", type=int, default=16, help="head num")
    parser.add_argument("--bos_id", type=int, default=2, help="bos id")
    parser.add_argument("--eos_id", type=int, default=2, help="eos id")
    parser.add_argument("--pad_id", type=int, default=1, help="pad id")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # export
    export_native_fs_transformer(
        args.output,
        args.input,
        args,
    )
    # test
    model_path = None
    if os.path.exists(args.output):
        model_path = args.output
    elif os.path.exists(args.output.replace("pb", "hdf5")):
        model_path = args.output.replace("pb", "hdf5")
    if model_path:
        ls_model = lsi.Transformer(model_path, 8)
        # Note that pad_id (1) should be on the right side
        # wmt14 en2de translation
        src_tokens = np.array([[10827, 27, 2081, 2], [1478, 6, 2, 1]])
        trg_out = ls_model.infer(src_tokens)
        print(trg_out)
        # reference result:
        # [[[ 4112,  6176,    12,  3380,     2],
        # [  399,  7572,    12,  3380,     2],
        # [ 4112,  6176,    12, 18882,     2],
        # [  740,  3380,  4331,     2,     2]],

        # [[ 1515,     5,     2,     2,     2],
        # [   71,  1515,     5,     2,     2],
        # [   22,  1515,     5,     2,     2],
        # [    5,     2,     2,     2,     2]]]
