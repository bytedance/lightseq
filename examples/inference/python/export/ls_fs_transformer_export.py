import torch
from proto.transformer_pb2 import Transformer
from lightseq.training import (
    export_ls_config,
    export_ls_embedding,
    export_ls_encoder,
    export_ls_decoder,
)
import lightseq.inference as lsi


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
    transformer.src_embedding.norm_scale[:] = (
        state_dict["encoder.layer_norm.weight"].flatten().tolist()
    )
    transformer.src_embedding.norm_bias[:] = (
        state_dict["encoder.layer_norm.bias"].flatten().tolist()
    )
    transformer.trg_embedding.norm_scale[:] = (
        state_dict["decoder.layer_norm.weight"].flatten().tolist()
    )
    transformer.trg_embedding.norm_bias[:] = (
        state_dict["decoder.layer_norm.bias"].flatten().tolist()
    )
    transformer.trg_embedding.shared_bias[:] = (
        torch.zeros(state_dict["decoder.embed_tokens.embeddings"].size(0))
        .flatten()
        .tolist()
    )


def export_ls_fs_transformer(ckpt_path, pb_path):
    with open(ckpt_path, "rb") as fin:
        ckpt_file = torch.load(fin)
    args = ckpt_file["args"]
    state_dict = ckpt_file["model"]

    transformer = Transformer()
    encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
    export_ls_embedding(transformer, encoder_state_dict, 1024, True)
    export_ls_embedding(transformer, decoder_state_dict, 1024, False)
    export_ls_encoder(
        transformer,
        encoder_state_dict,
        args.encoder_embed_dim,
        args.encoder_ffn_embed_dim,
    )
    export_ls_decoder(
        transformer,
        decoder_state_dict,
        args.decoder_embed_dim,
        args.decoder_ffn_embed_dim,
        args.decoder_layers,
    )
    export_fs_weights(transformer, state_dict)
    export_ls_config(transformer, args.encoder_attention_heads, 2, 2, 6)

    with open(pb_path, "wb") as fout:
        fout.write(transformer.SerializeToString())


if __name__ == "__main__":
    ckpt_path = "checkpoints/checkpoint_best.pt"
    pb_path = "checkpoints/transformer.pb"
    export_ls_fs_transformer(ckpt_path, pb_path)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2]]
    model = lsi.Transformer(pb_path, 128)
    output = model.infer(src)
    print(output)
