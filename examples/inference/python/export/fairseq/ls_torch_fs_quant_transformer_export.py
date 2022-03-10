"""
Export quantized Fairseq Transformer models training using custom Torch layers to protobuf/hdf5 format.
Refer to the `examples/training/fairseq` directory for more training details.
"""
from collections import OrderedDict
import argparse

import torch
import tensorflow as tf
from export.proto.quant_transformer_pb2 import QuantTransformer
from lightseq.training.ops.pytorch.export import export_ls_config, apply_rule
from lightseq.training.ops.pytorch.export_ptq import (
    gather_quant_token_embedding,
    quantize,
)
from lightseq.training.ops.pytorch.util import get_pos_embedding
import lightseq.inference as lsi


enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "self_attn_layer_norm weight",
        "multihead_norm_bias": "self_attn_layer_norm bias",
        "multihead_project_kernel_qkv": "self_attn qkv_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn qkv_proj bias",
        "multihead_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
        # weight_clip_max
        "multihead_project_kernel_qkv_clip_max": "self_attn qkv_proj weight_quant clip_value_max",
        "multihead_project_kernel_output_clip_max": "self_attn out_proj weight_quant clip_value_max",
        "ffn_first_kernel_clip_max": "fc1 weight_quant clip_value_max",
        "ffn_second_kernel_clip_max": "fc2 weight_quant clip_value_max",
        # act_clip_max
        "multihead_ln_clip_max": "self_attn qkv_proj input_quant clip_value_max",
        "multihead_project_output_clip_max": "self_attn out_proj input_quant clip_value_max",
        "ffn_ln_clip_max": "fc1 input_quant clip_value_max",
        "ffn_first_act_clip_max": "fc2 input_quant clip_value_max",
        "multihead_qkv_dense_clip_max": "self_attn qkv_proj output_quant clip_value_max",
        "multihead_output_dense_clip_max": "self_attn out_proj output_quant clip_value_max",
        "ffn_first_output_clip_max": "fc1 output_quant clip_value_max",
    }
)

dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "self_attn_layer_norm weight",
        "self_norm_bias": "self_attn_layer_norm bias",
        "self_project_kernel_qkv": "self_attn qkv_proj weight&&expression_.transpose(0, 1)",
        "self_project_bias_qkv": "self_attn qkv_proj bias",
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
        # weight_clip_max
        "self_project_kernel_qkv_clip_max": "self_attn qkv_proj weight_quant clip_value_max",
        "self_project_kernel_output_clip_max": "self_attn out_proj weight_quant clip_value_max",
        "encdec_project_kernel_q_clip_max": "encoder_attn q_proj weight_quant clip_value_max",
        "encdec_project_kernel_output_clip_max": "encoder_attn out_proj weight_quant clip_value_max",
        "ffn_first_kernel_clip_max": "fc1 weight_quant clip_value_max",
        "ffn_second_kernel_clip_max": "fc2 weight_quant clip_value_max",
        # act_clip_max
        "self_ln_clip_max": "self_attn qkv_proj input_quant clip_value_max",
        "self_project_output_clip_max": "self_attn out_proj input_quant clip_value_max",
        "encdec_ln_clip_max": "encoder_attn q_proj input_quant clip_value_max",
        "encdec_project_output_clip_max": "encoder_attn out_proj input_quant clip_value_max",
        "ffn_ln_clip_max": "fc1 input_quant clip_value_max",
        "ffn_first_act_clip_max": "fc2 input_quant clip_value_max",
        "self_qkv_dense_clip_max": "self_attn qkv_proj output_quant clip_value_max",
        "self_output_dense_clip_max": "self_attn out_proj output_quant clip_value_max",
        "encdec_q_dense_clip_max": "encoder_attn q_proj output_quant clip_value_max",
        "encdec_output_dense_clip_max": "encoder_attn out_proj output_quant clip_value_max",
        "ffn_first_output_clip_max": "fc1 output_quant clip_value_max",
        "self_qkv_bias_out_clip_max": "self_attn attention_quant clip_value_max",
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
    encode_output_kernel_clip_max_pattern = [
        "encoder_attn {0} k_proj weight_quant clip_value_max".format(ele)
        for ele in range(dec_layer_num)
    ]
    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
        "encode_output_project_kernel_kv_clip_max": "&&".join(
            encode_output_kernel_clip_max_pattern
        ),
        "output_ln_clip_max": "output_projection input_quant clip_value_max",
        "logits_clip_max": "output_projection output_quant clip_value_max",
    }


def fill_quant_pb_layer(tensor_names, state_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        if proto_name.endswith("_clip_max"):
            if proto_name == "encode_output_project_kernel_kv_clip_max":
                exec("layer.%s[:]=target_tensor" % proto_name)
            else:
                target_value = float(target_tensor[0])
                exec("layer.%s=target_value" % proto_name)
        elif "kernel" in proto_name:
            exec("layer.%s=bytes(target_tensor.flatten().tolist())" % proto_name)
        else:
            exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def export_ls_torch_fs_quant_transformer(
    model_dir,
    pb_path,
    max_step=512,
    bos_id=2,
    eos_id=2,
    pad_id=1,
):
    transformer = QuantTransformer()
    # load var names
    reloaded = torch.load(model_dir, "cpu")
    args = reloaded["args"]
    model_dict = reloaded["model"]

    var_names = list(model_dict.keys())
    for name in var_names:
        if name.endswith("weight_quant.clip.clip_value_max"):
            model_dict[name[:-26]] = torch.Tensor(
                quantize(model_dict[name[:-26]].numpy(), 127, model_dict[name].numpy())
            ).int()

    trg_emb_mapping_dict["shared_bias"] = (
        "expression_np.zeros(%d)"
        % model_dict["decoder.embed_tokens.emb_lookup.weight"].shape[0]
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
        fill_quant_pb_layer(
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
        fill_quant_pb_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    fill_quant_pb_layer(
        enc_var_name_list,
        encoder_state_dict,
        transformer.src_embedding,
        src_emb_mapping_dict,
    )

    # encoder token embedding
    src_tb_clip_max = model_dict[
        "encoder.embed_tokens.emb_quant.clip.clip_value_max"
    ].numpy()
    src_tb, _, _ = gather_quant_token_embedding(
        enc_var_name_list, encoder_state_dict, "emb_lookup", src_tb_clip_max
    )
    transformer.src_embedding.token_embedding = bytes(src_tb.flatten().tolist())
    transformer.src_embedding.emb_clip_max = src_tb_clip_max

    # encoder position embedding
    pos_emb = None
    if "encoder.embed_tokens.embed_positions.weight" in encoder_state_dict:
        pos_emb = encoder_state_dict[
            "encoder.embed_tokens.embed_positions.weight"
        ].numpy()
        transformer.src_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = get_pos_embedding(max_step, src_tb.shape[-1]).numpy()
        transformer.src_embedding.position_embedding[:] = pos_emb.flatten().tolist()

    print(
        "encoder.embed_tokens.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
            pos_emb.shape
        )
    )

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    trg_emb_mapping_dict.update(encode_output_mapping_dict)
    fill_quant_pb_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        trg_emb_mapping_dict,
    )
    # decoder token embedding
    trg_tb_clip_max = model_dict[
        "decoder.embed_tokens.emb_quant.clip.clip_value_max"
    ].numpy()
    trg_tb, _, _ = gather_quant_token_embedding(
        dec_var_name_list, decoder_state_dict, "emb_lookup", trg_tb_clip_max
    )
    transformer.trg_embedding.token_embedding = bytes(
        trg_tb.transpose().flatten().tolist()
    )
    transformer.trg_embedding.emb_clip_max = trg_tb_clip_max
    print(
        "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            trg_tb.transpose().shape
        )
    )
    # decoder position embedding
    pos_emb = None
    if "decoder.embed_tokens.embed_positions.weight" in decoder_state_dict:
        pos_emb = decoder_state_dict[
            "decoder.embed_tokens.embed_positions.weight"
        ].numpy()
        transformer.trg_embedding.position_embedding[:] = pos_emb.flatten().tolist()
    else:
        pos_emb = get_pos_embedding(max_step, trg_tb.shape[-1]).numpy()
        transformer.trg_embedding.position_embedding[:] = pos_emb.flatten().tolist()

    print(
        "decoder.embed_tokens.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
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
    pb_path = "quant_transformer.pb"
    export_ls_torch_fs_quant_transformer(args.model, pb_path)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1, 1, 1]]
    pb_model = lsi.QuantTransformer(pb_path, 8)
    pb_output = pb_model.infer(src)
    # Expected result: [23, 550, 34, 118, 148, 2939, 4, 42, 32, 37, 6, 224, 10, 179, 5, 2]
    print("pb results:", pb_output)
