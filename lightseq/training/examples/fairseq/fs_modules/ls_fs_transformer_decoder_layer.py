from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
import torch.nn.functional as F
from lightseq.training.ops.pytorch.transformer_decoder_layer import (
    LSTransformerDecoderLayer,
)


@with_incremental_state
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        static_kv=False,
        attn_qw=None,
        attn_qb=None,
        attn_kw=None,
        attn_kb=None,
        attn_vw=None,
        attn_vb=None,
        attn_ow=None,
        attn_ob=None,
    ):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self.get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = F.linear(query, attn_qw, attn_qb)
            k = F.linear(query, attn_kw, attn_kb)
            v = F.linear(query, attn_vw, attn_vb)
        elif self.encoder_decoder_attention:
            q = F.linear(query, attn_qw, attn_qb)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = F.linear(key, attn_kw, attn_kb)
                v = F.linear(key, attn_vw, attn_vb)
        else:
            assert key is not None and value is not None
            q = F.linear(query, attn_qw, attn_qb)
            k = F.linear(key, attn_kw, attn_kb)
            v = F.linear(value, attn_vw, attn_vb)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = F.linear(attn, attn_ow, attn_ob)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self.get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class LSFSTransformerDecoderLayer(LSTransformerDecoderLayer):
    """Decoder layer only for inference."""

    def __init__(self, config, initial_weights=None, initial_biases=None):

        super().__init__(config, initial_weights, initial_biases)
        self.embed_dim = self.config.hidden_size

        self.self_attn = self.build_self_attention(self.embed_dim, self.config.nhead)

        self.activation_fn = utils.get_activation_fn(activation="relu")
        self.normalize_before = self.config.pre_layer_norm

        self.encodec_attn = self.build_encoder_attention(
            self.embed_dim, self.config.nhead
        )

        self.eps = 1e-8
        self.has_weights = False

    def build_self_attention(self, embed_dim, nhead):
        return MultiheadAttention(embed_dim, nhead, self_attention=True)

    def build_encoder_attention(self, embed_dim, nhead):
        return MultiheadAttention(embed_dim, nhead, encoder_decoder_attention=True)

    def extract_weights(self):
        if self.has_weights:
            return
        self.has_weights = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
    ):
        if incremental_state is None:
            cache = None
            res = super().forward(x, encoder_out, encoder_padding_mask, cache)
            return res, None, None
        # predict
        (
            attn_qkvw,
            attn_qkvb,
            attn_ow,
            attn_ob,
            attn_nw,
            attn_nb,
            encdec_attn_qw,
            encdec_attn_qb,
            encdec_attn_ow,
            encdec_attn_ob,
            encdec_attn_nw,
            encdec_attn_nb,
            inter_w,
            inter_b,
            output_w,
            output_b,
            ffn_nw,
            ffn_nb,
            encdec_attn_kvw,
            encdec_attn_kvb,
        ) = self.split_weights()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1).contiguous()
        residual = x
        if self.normalize_before:
            x = F.layer_norm(
                x, tuple((self.embed_dim,)), attn_nw, attn_nb, eps=self.eps
            )
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            attn_qw=attn_qkvw[0 : self.embed_dim],
            attn_qb=attn_qkvb[0 : self.embed_dim],
            attn_kw=attn_qkvw[self.embed_dim : 2 * self.embed_dim],
            attn_kb=attn_qkvb[self.embed_dim : 2 * self.embed_dim],
            attn_vw=attn_qkvw[2 * self.embed_dim : 3 * self.embed_dim],
            attn_vb=attn_qkvb[2 * self.embed_dim : 3 * self.embed_dim],
            attn_ow=attn_ow,
            attn_ob=attn_ob,
        )
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = F.layer_norm(
                x, tuple((self.embed_dim,)), attn_nw, attn_nb, eps=self.eps
            )

        residual = x
        if self.normalize_before:
            x = F.layer_norm(
                x,
                tuple((self.embed_dim,)),
                encdec_attn_nw,
                encdec_attn_nb,
                eps=self.eps,
            )
        x, _ = self.encodec_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            attn_qw=encdec_attn_qw,
            attn_qb=encdec_attn_qb,
            attn_kw=encdec_attn_kvw[0 : self.embed_dim],
            attn_kb=encdec_attn_kvb[0 : self.embed_dim],
            attn_vw=encdec_attn_kvw[self.embed_dim : 2 * self.embed_dim],
            attn_vb=encdec_attn_kvb[self.embed_dim : 2 * self.embed_dim],
            attn_ow=encdec_attn_ow,
            attn_ob=encdec_attn_ob,
        )
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = F.layer_norm(
                x,
                tuple((self.embed_dim,)),
                encdec_attn_nw,
                encdec_attn_nb,
                eps=self.eps,
            )

        residual = x
        if self.normalize_before:
            x = F.layer_norm(x, tuple((self.embed_dim,)), ffn_nw, ffn_nb, eps=self.eps)
        x = F.linear(x, inter_w, inter_b)
        x = self.activation_fn(x)
        x = F.linear(x, output_w, output_b)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = F.layer_norm(x, tuple((self.embed_dim,)), ffn_nw, ffn_nb, eps=self.eps)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1).contiguous()
        return x, None, None
