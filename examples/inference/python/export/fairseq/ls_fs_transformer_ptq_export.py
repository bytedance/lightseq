"""
Export Fairseq Transformer models training with LightSeq modules
to int8 protobuf format using post training quantization.
Refer to the `examples/training/fairseq` directory for more training details.
"""
import torch
from export.proto.quant_transformer_pb2 import QuantTransformer
from lightseq.training import (
    export_ls_config,
    export_ls_embedding_ptq,
    export_ls_encoder_ptq,
    export_ls_decoder_ptq,
)
import lightseq.inference as lsi
from export.util import parse_args, save_model


# adjust this value to achieve better performance
global_act_clip_max = 45.0


def _extract_weight(state_dict):
    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in state_dict:
        if k.startswith("encoder."):
            encoder_state_dict[k] = state_dict[k]
        if k.startswith("decoder."):
            decoder_state_dict[k] = state_dict[k]
    return encoder_state_dict, decoder_state_dict


def export_fs_weights(transformer, state_dict):
    enc_norm_w = state_dict["encoder.layer_norm.weight"].flatten().tolist()
    enc_norm_b = state_dict["encoder.layer_norm.bias"].flatten().tolist()
    dec_norm_w = state_dict["decoder.layer_norm.weight"].flatten().tolist()
    dec_norm_b = state_dict["decoder.layer_norm.bias"].flatten().tolist()
    dec_shared_b = (
        torch.zeros(state_dict["decoder.embed_tokens.embeddings"].size(0))
        .flatten()
        .tolist()
    )
    transformer.src_embedding.norm_scale[:] = enc_norm_w
    transformer.src_embedding.norm_bias[:] = enc_norm_b
    transformer.trg_embedding.norm_scale[:] = dec_norm_w
    transformer.trg_embedding.norm_bias[:] = dec_norm_b
    transformer.trg_embedding.shared_bias[:] = dec_shared_b


def export_ls_fs_transformer_ptq(model_path, pb_path, hdf5_path, hdf5):
    with open(model_path, "rb") as fin:
        ckpt_file = torch.load(fin)
    args = ckpt_file["args"]
    state_dict = ckpt_file["model"]

    transformer = QuantTransformer()
    encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
    export_ls_embedding_ptq(
        transformer,
        encoder_state_dict,
        300,
        True,
        save_pb=True,
    )
    export_ls_embedding_ptq(
        transformer,
        decoder_state_dict,
        300,
        False,
        save_pb=True,
    )
    export_ls_encoder_ptq(
        transformer,
        encoder_state_dict,
        args.encoder_embed_dim,
        args.encoder_ffn_embed_dim,
        act_clip_max=global_act_clip_max,
        save_pb=True,
    )
    export_ls_decoder_ptq(
        transformer,
        decoder_state_dict,
        args.decoder_embed_dim,
        args.decoder_ffn_embed_dim,
        args.decoder_layers,
        act_clip_max=global_act_clip_max,
        save_pb=True,
    )
    export_fs_weights(transformer, state_dict)
    export_ls_config(
        transformer,
        args.encoder_attention_heads,
        1,
        2,
        2,
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
    path = export_ls_fs_transformer_ptq(args.model, pb_path, hdf5_path, args.hdf5)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1, 1, 1]]
    model = lsi.QuantTransformer(path, 8)
    output = model.infer(src)
    # Expected result: [23, 550, 34, 118, 148, 2939, 4, 42, 32, 37, 6, 224, 10, 179, 5, 2]
    print("results:", output)
