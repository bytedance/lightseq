"""
Export Fairseq Transformer models training with LightSeq to protobuf format,
and then using int8 quantization to speedup inference.
Refer to the `examples/training/fairseq` directory for more training details.
"""
import torch
import h5py
from proto.quant_transformer_pb2 import QuantTransformer
from lightseq.training import (
    export_ls_config,
    export_ls_embedding_ptq,
    export_ls_encoder_ptq,
    export_ls_decoder_ptq,
)
import lightseq.inference as lsi


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


def export_fs_weights(file, state_dict, save_pb=True):
    enc_norm_w = state_dict["encoder.layer_norm.weight"].flatten().tolist()
    enc_norm_b = state_dict["encoder.layer_norm.bias"].flatten().tolist()
    dec_norm_w = state_dict["decoder.layer_norm.weight"].flatten().tolist()
    dec_norm_b = state_dict["decoder.layer_norm.bias"].flatten().tolist()
    dec_shared_b = (
        torch.zeros(state_dict["decoder.embed_tokens.embeddings"].size(0))
        .flatten()
        .tolist()
    )
    file.src_embedding.norm_scale[:] = enc_norm_w
    file.src_embedding.norm_bias[:] = enc_norm_b
    file.trg_embedding.norm_scale[:] = dec_norm_w
    file.trg_embedding.norm_bias[:] = dec_norm_b
    file.trg_embedding.shared_bias[:] = dec_shared_b


def export_ls_fs_transformer(ckpt_path, out_path, save_pb=True):
    with open(ckpt_path, "rb") as fin:
        ckpt_file = torch.load(fin)
    args = ckpt_file["args"]
    state_dict = ckpt_file["model"]

    file = QuantTransformer()
    encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
    export_ls_embedding_ptq(
        file,
        encoder_state_dict,
        1024,
        True,
        clip_max=global_act_clip_max,
        save_pb=save_pb,
    )
    export_ls_embedding_ptq(
        file,
        decoder_state_dict,
        1024,
        False,
        clip_max=global_act_clip_max,
        save_pb=save_pb,
    )
    export_ls_encoder_ptq(
        file,
        encoder_state_dict,
        args.encoder_embed_dim,
        args.encoder_ffn_embed_dim,
        act_clip_max=global_act_clip_max,
        save_pb=save_pb,
    )
    export_ls_decoder_ptq(
        file,
        decoder_state_dict,
        args.decoder_embed_dim,
        args.decoder_ffn_embed_dim,
        args.decoder_layers,
        act_clip_max=global_act_clip_max,
        save_pb=save_pb,
    )
    export_fs_weights(file, state_dict, save_pb)
    export_ls_config(
        file,
        args.encoder_attention_heads,
        2,
        2,
        6,
        args.encoder_layers,
        args.decoder_layers,
        save_pb=save_pb,
    )

    with open(out_path, "wb") as fout:
        fout.write(file.SerializeToString())


if __name__ == "__main__":
    ckpt_path = "checkpoint_best.pt"
    pb_path = "quant_transformer.pb"
    print("export to pb model >>>>>>")
    export_ls_fs_transformer(ckpt_path, pb_path)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2]]
    pb_model = lsi.QuantTransformer(pb_path, 8)
    pb_output = pb_model.infer(src)
    # FP16 result: [23, 550, 34, 118, 148, 2939, 4, 42, 32, 37, 6]
    print("pb results:", pb_output)
