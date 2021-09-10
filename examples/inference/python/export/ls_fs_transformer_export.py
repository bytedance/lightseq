"""
Export Fairseq Transformer models training with LightSeq to protobuf/hdf5 format.
Refer to the `examples/training/fairseq` directory for more training details.
"""
import torch
import h5py
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
    if save_pb:
        file.src_embedding.norm_scale[:] = enc_norm_w
        file.src_embedding.norm_bias[:] = enc_norm_b
        file.trg_embedding.norm_scale[:] = dec_norm_w
        file.trg_embedding.norm_bias[:] = dec_norm_b
        file.trg_embedding.shared_bias[:] = dec_shared_b
    else:
        file.create_dataset("src_embedding/norm_scale", data=enc_norm_w, dtype="f4")
        file.create_dataset("src_embedding/norm_bias", data=enc_norm_b, dtype="f4")
        file.create_dataset("trg_embedding/norm_scale", data=dec_norm_w, dtype="f4")
        file.create_dataset("trg_embedding/norm_bias", data=dec_norm_b, dtype="f4")
        file.create_dataset("trg_embedding/shared_bias", data=dec_shared_b, dtype="f4")


def export_ls_fs_transformer(ckpt_path, out_path, save_pb=True):
    with open(ckpt_path, "rb") as fin:
        ckpt_file = torch.load(fin)
    args = ckpt_file["args"]
    state_dict = ckpt_file["model"]

    if save_pb:
        file = Transformer()
    else:
        file = h5py.File(out_path, "w")
    encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
    export_ls_embedding(file, encoder_state_dict, 1024, True, save_pb)
    export_ls_embedding(file, decoder_state_dict, 1024, False, save_pb)
    export_ls_encoder(
        file,
        encoder_state_dict,
        args.encoder_embed_dim,
        args.encoder_ffn_embed_dim,
        save_pb,
    )
    export_ls_decoder(
        file,
        decoder_state_dict,
        args.decoder_embed_dim,
        args.decoder_ffn_embed_dim,
        args.decoder_layers,
        save_pb,
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

    if save_pb:
        with open(out_path, "wb") as fout:
            fout.write(file.SerializeToString())
    else:
        file.close()


if __name__ == "__main__":
    ckpt_path = "checkpoints/checkpoint_best.pt"
    pb_path = "checkpoints/transformer.pb"
    hdf5_path = "checkpoints/transformer.hdf5"
    print("export to pb model >>>>>>")
    export_ls_fs_transformer(ckpt_path, pb_path)
    print("export to hdf5 model >>>>>>")
    export_ls_fs_transformer(ckpt_path, hdf5_path, save_pb=False)
    src = [[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2]]
    pb_model = lsi.Transformer(pb_path, 8)
    pb_output = pb_model.infer(src)
    hdf5_model = lsi.Transformer(hdf5_path, 8)
    hdf5_output = hdf5_model.infer(src)
    print("pb results:", pb_output)
    print("hdf5 results:", hdf5_output)
