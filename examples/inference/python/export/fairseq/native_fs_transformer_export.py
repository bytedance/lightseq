"""
Export native Fairseq Transformer models to protobuf/hdf5 format.
Refer to the `examples/training/fairseq` directory for more training details.
"""
from collections import OrderedDict
import argparse

import torch
import tensorflow as tf
import h5py
from export.proto.transformer_pb2 import Transformer
from lightseq.training.ops.pytorch.export import (
    gather_token_embedding,
    fill_pb_layer,
    export_ls_config,
    export_pb2hdf5,
)
from lightseq.training.ops.pytorch.util import get_pos_embedding
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


def export_native_fs_transformer(
    model_dir,
    pb_path,
    hdf5_path,
    max_step=512,
    bos_id=2,
    eos_id=2,
    pad_id=1,
):
    transformer = Transformer()
    # load var names
    reloaded = torch.load(model_dir, "cpu")
    args = reloaded["args"]
    model_dict = reloaded["model"]

    trg_emb_mapping_dict["shared_bias"] = (
        "expression_np.zeros(%d)"
        % model_dict["decoder.embed_tokens.weight"].numpy().shape[0]
    )

    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in model_dict:
        if k.startswith("encoder."):
            encoder_state_dict[k] = model_dict[k]
        if k.startswith("decoder."):
            decoder_state_dict[k] = model_dict[k]

    dec_var_name_list = list(decoder_state_dict.keys())
    enc_var_name_list = list(encoder_state_dict.keys())

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
        pos_emb = get_pos_embedding(max_step + pad_id + 1, src_tb.shape[-1]).numpy()
        pos_emb_list = (
            pos_emb[pad_id + 1 : max_step + pad_id + 1, :].reshape([-1]).tolist()
        )
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
        pos_emb = get_pos_embedding(max_step + pad_id + 1, trg_tb.shape[-1]).numpy()
        pos_emb_list = (
            pos_emb[pad_id + 1 : max_step + pad_id + 1, :].reshape([-1]).tolist()
        )
        transformer.trg_embedding.position_embedding[:] = pos_emb_list

    print(
        "decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill in conf
    export_ls_config(
        transformer,
        args.encoder_attention_heads,
        pad_id,
        bos_id,
        eos_id,
        args.encoder_layers,
        args.decoder_layers,
        save_pb=True,
    )

    print("Writing to {0}".format(pb_path))
    with tf.io.gfile.GFile(pb_path, "wb") as fout:
        fout.write(transformer.SerializeToString())

    print("Writing to {0}".format(hdf5_path))
    f = h5py.File(hdf5_path, "w")
    export_pb2hdf5(transformer, f)
    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="export fairseq checkpoint", usage="")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="checkpoint_best.pt",
        help="path of fairseq checkpoint",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    pb_path = "transformer.pb"
    hdf5_path = "transformer.hdf5"
    export_native_fs_transformer(args.model, pb_path, hdf5_path)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1, 1, 1]]
    pb_model = lsi.Transformer(pb_path, 8)
    pb_output = pb_model.infer(src)
    # Expected result: [23, 550, 34, 118, 148, 2939, 4, 42, 32, 37, 6, 224, 10, 179, 5, 2]
    print("pb results:", pb_output)
