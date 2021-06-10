"""
Copyright 2021 The LightSeq Team
Copyright Facebook Fairseq
We use layers from Facebook Fairseq as our baseline for unit test
"""

from typing import Dict, List, Optional
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """Encoder layer implemented by fairseq.
    This version only removes the "args" parameter, no other changes

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    normalize_before to True.
    """

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        nhead,
        dropout,
        attn_dropout,
        activation_dropout,
        normalize_before=True,
        activation_fn="relu",
        quant_noise=0,
        quant_noise_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_noise = quant_noise
        self.quant_noise_block_size = quant_noise_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, nhead, attn_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=activation_fn)
        activation_dropout_p = activation_dropout
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, nhead, attn_dropout):
        return MultiheadAttention(
            embed_dim,
            nhead,
            dropout=attn_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))

        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer implemented by fairseq.
    This version only removes the "args" parameter, no other changes
    """

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        nhead,
        encoder_embed_dim,
        dropout,
        attn_dropout,
        activation_dropout,
        normalize_before=True,
        activation_fn="relu",
        quant_noise=0,
        quant_noise_block_size=8,
        cross_self_attention=False,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = quant_noise
        self.quant_noise_block_size = quant_noise_block_size

        self.cross_self_attention = cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            nhead,
            attn_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(activation=activation_fn)
        activation_dropout_p = activation_dropout
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = normalize_before

        export = False
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encodec_attn = None
            self.encodec_attn_layer_norm = None
        else:
            self.encodec_attn = self.build_encoder_attention(
                self.embed_dim, encoder_embed_dim, attn_dropout, nhead
            )
            self.encodec_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, nhead, attn_dropout, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            nhead,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(
        self, embed_dim, encoder_embed_dim, attn_dropout, nhead
    ):
        return MultiheadAttention(
            embed_dim,
            nhead,
            kdim=encoder_embed_dim,
            vdim=encoder_embed_dim,
            dropout=attn_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encodec_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encodec_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encodec_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encodec_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encodec_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def generate_enc_layer():
    hidden_size = 1024
    intermediate_size = 1024 * 4
    heads = 16
    hidden_dropout_ratio = 0.0
    attn_dropout_ratio = 0.0
    activation_dropout_ratio = 0.0
    pre_layer_norm = True
    layer = TransformerEncoderLayer(
        hidden_size,
        intermediate_size,
        heads,
        hidden_dropout_ratio,
        attn_dropout_ratio,
        activation_dropout_ratio,
        pre_layer_norm,
        activation_fn="relu",
    )
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


def generate_dec_layer():
    hidden_size = 1024
    intermediate_size = 1024 * 4
    heads = 16
    hidden_dropout_ratio = 0.0
    attn_dropout_ratio = 0.0
    activation_dropout_ratio = 0.0
    pre_layer_norm = True
    layer = TransformerDecoderLayer(
        embed_dim=hidden_size,
        ffn_embed_dim=intermediate_size,
        nhead=heads,
        encoder_embed_dim=hidden_size,
        dropout=hidden_dropout_ratio,
        attn_dropout=attn_dropout_ratio,
        activation_dropout=activation_dropout_ratio,
        normalize_before=pre_layer_norm,
        activation_fn="relu",
    )

    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024, fp16=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        ).to(torch.device("cuda:0"))
        if fp16:
            self.weights = self.weights.to(torch.half)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def make_positions(self, tensor):
        mask = torch.ones_like(tensor)
        return (torch.cumsum(mask, dim=1).type_as(mask) - 1).long()

    def forward(
        self,
        input,
        incremental_state=None,
        timestep=None,
        positions=None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size(0), input.size(1)
        positions = self.make_positions(input)
        mask = (
            torch.ne(input, self.padding_idx)
            .unsqueeze(2)
            .expand(bsz, seq_len, self.embedding_dim)
        )
        return (
            self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)
            * mask
        ).detach()


class TransformerEmbeddingLayer(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, max_seq_len, padding_idx, dropout, fp16
    ):
        super().__init__()

        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        nn.init.normal_(self.embeddings.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.embeddings.weight[padding_idx], 0)
        self.embeddings.to(
            torch.device("cuda:0"), dtype=(torch.half if fp16 else torch.float)
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, max_seq_len, fp16
        ).to(torch.device("cuda:0"))
        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def forward(self, input):
        x = self.embeddings(input)
        x = math.sqrt(self.embedding_dim) * x
        x += self.embed_positions(input)
        x = F.dropout(x, p=self.dropout, training=True)

        return x


def generate_emb_layer(ls_emb_config):
    layer = TransformerEmbeddingLayer(
        ls_emb_config.vocab_size,
        ls_emb_config.embedding_dim,
        ls_emb_config.max_seq_len,
        ls_emb_config.padding_idx,
        ls_emb_config.dropout,
        ls_emb_config.fp16,
    )
    dtype = torch.float16 if ls_emb_config.fp16 else torch.float32
    layer.to(torch.device("cuda:0"), dtype=dtype)

    return layer


if __name__ == "__main__":
    generate_enc_layer()
    generate_dec_layer()
