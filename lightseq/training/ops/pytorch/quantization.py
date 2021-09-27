import torch
from lightseq.training import LSTransformerEncoderLayer, LSTransformerDecoderLayer


def quantize_tensor(x, scale, clip_max):
    x_int8 = torch.clamp(x / clip_max * scale, min=-scale, max=scale).to(
        dtype=torch.int8
    )
    return x_int8


def quantize_ls_embedding(state_dict):
    for wn in state_dict.keys():
        if "embeddings" in wn:
            # TODO: replace 0.5 by the learned scale of QAT
            state_dict[wn] = quantize_tensor(state_dict[wn], 127, 0.5)


def quantize_ls_encoder(state_dict, hidden_size, intermediate_size):
    hs, ims = hidden_size, intermediate_size
    offsets = LSTransformerEncoderLayer.gen_offset(hs, ims)
    # quantize 0, 2, 6, 8
    for wn in state_dict.keys():
        if "para" in wn:
            # TODO: replace 0.5 by the learned scale of QAT
            state_dict[wn] = quantize_tensor(
                state_dict[wn][offsets[0] : offsets[1]], 127, 0.5
            )
