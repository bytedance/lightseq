"""
Export native Fairseq Transformer models to int8 protobuf format using post training quantization.
Refer to the `examples/training/fairseq` directory for more training details.
"""
from collections import OrderedDict

import torch
from export.proto.quant_transformer_pb2 import QuantTransformer
from lightseq.training.ops.pytorch.export import export_ls_config
from lightseq.training.ops.pytorch.export_quant import (
    gather_quant_token_embedding,
    fill_quant_pb_layer,
)
from lightseq.training.ops.pytorch.util import get_pos_embedding
import lightseq.inference as lsi
from export.fairseq.util import parse_args, save_model


# adjust this value to achieve better performance
global_act_clip_max = 45.0


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
        # act_clip_max
        "multihead_ln_clip_max": "None",
        "multihead_project_output_clip_max": "None",
        "ffn_ln_clip_max": "None",
        "ffn_first_act_clip_max": "None",
        "multihead_qkv_dense_clip_max": "None",
        "multihead_output_dense_clip_max": "None",
        "ffn_first_output_clip_max": "None",
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
        # act_clip_max
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

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
        "output_ln_clip_max": "output_projection input_quant clip_value_max",
        "logits_clip_max": "output_projection output_quant clip_value_max",
    }


def export_native_fs_transformer(
    model_path,
    pb_path,
    hdf5_path,
    hdf5,
    max_step=300,
    bos_id=2,
    eos_id=2,
    pad_id=1,
):
    transformer = QuantTransformer()
    # load var names
    reloaded = torch.load(model_path, "cpu")
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
        fill_quant_pb_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            transformer.encoder_stack.add(),
            enc_layer_mapping_dict,
            global_act_clip_max,
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
            global_act_clip_max,
        )

    fill_quant_pb_layer(
        enc_var_name_list,
        encoder_state_dict,
        transformer.src_embedding,
        src_emb_mapping_dict,
        global_act_clip_max,
    )
    # encoder token embedding
    src_tb, src_tb_clip_max, _ = gather_quant_token_embedding(
        enc_var_name_list, encoder_state_dict, "embed_tokens"
    )
    transformer.src_embedding.token_embedding = bytes(src_tb.flatten().tolist())
    transformer.src_embedding.emb_clip_max = src_tb_clip_max
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
    fill_quant_pb_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        trg_emb_mapping_dict,
        global_act_clip_max,
        len(dec_tensor_names.keys()),
    )
    # decoder token embedding
    trg_tb, trg_tb_clip_max, _ = gather_quant_token_embedding(
        dec_var_name_list, decoder_state_dict, "embed_tokens"
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

    save_path = save_model(transformer, pb_path, hdf5_path, hdf5)
    return save_path


if __name__ == "__main__":
    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    pb_path = f"{model_name}_ptq.pb"
    hdf5_path = f"{model_name}_ptq.hdf5"
    path = export_native_fs_transformer(args.model, pb_path, hdf5_path, args.hdf5)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1, 1, 1]]
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1, 1, 1]]
    model = lsi.QuantTransformer(path, 8)
    output = model.infer(src)
    # Expected result: [23, 550, 34, 118, 148, 2939, 4, 42, 32, 37, 6, 224, 10, 179, 5, 2]
    print("results:", output)
