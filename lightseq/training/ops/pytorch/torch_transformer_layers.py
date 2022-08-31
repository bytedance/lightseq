# Copyright 2021 The LightSeq Team
# Copyright Facebook Fairseq
# We use layers from Facebook Fairseq as our baseline


import math
import uuid

from typing import Dict, Optional, List

import torch
from torch import Tensor, nn
from torch.nn import Parameter, LayerNorm, Dropout

from lightseq.training.ops.pytorch import util
from lightseq.training.ops.pytorch.layer_base import (
    TransformerEmbeddingLayerBase,
    TransformerEncoderLayerBase,
    TransformerDecoderLayerBase,
)
from .quantization import (
    QuantLinear,
    TensorQuantizer,
    act_quant_config,
    weight_quant_config,
    out_quant_config,
    emb_quant_config,
)


def copy_para(x, fp16):
    y = util.copy_para(x)
    return y.half() if fp16 else y.float()


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        is_decoder=False,
        has_causal_mask=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.has_causal_mask = has_causal_mask

        self.num_heads = num_heads
        self.dropout_module = Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.is_decoder = is_decoder

        max_positions = 1024
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.uint8)
            ).view(1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        assert (
            not self.self_attention or self.qkv_same_dim
        ), "Self-attention requires query, key and value to be of the same size"

        self.attention_quant = None
        if self.self_attention:
            # self.qkv_proj = Linear(embed_dim, 3*embed_dim, bias=bias)
            self.qkv_proj = QuantLinear(embed_dim, 3 * embed_dim, bias=bias)

            self.attention_quant = (
                TensorQuantizer(out_quant_config) if self.is_decoder else None
            )
        elif self.encoder_decoder_attention and self.is_decoder:
            self.k_proj = QuantLinear(
                self.kdim, embed_dim, pre_activation="encoder_out", bias=bias
            )
            self.v_proj = QuantLinear(
                self.vdim, embed_dim, pre_activation="encoder_out", bias=bias
            )
            self.q_proj = QuantLinear(embed_dim, embed_dim, bias=bias)

        self.out_proj = QuantLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False
        self.init_incremental_state()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            if self.self_attention:
                nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
            else:
                nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            qkv = self.qkv_proj(query)
            if self.attention_quant is not None:
                qkv = self.attention_quant(qkv)
            q, k, v = qkv.split(self.embed_dim, dim=-1)
            # q = self.q_proj(query)
            # k = self.k_proj(query)
            # v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

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

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if self.has_causal_mask:
            query_length, key_length = query.size(0), key.size(0)
            causal_mask = self.bias[
                :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = util.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

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
        input_buffer = self._get_input_buffer(incremental_state)
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

    def _get_input_buffer(
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

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


class TransformerEncoderLayer(TransformerEncoderLayerBase):
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

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = self.build_self_attention(
            self.embed_dim, config.nhead, config.attn_prob_dropout_ratio
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = Dropout(config.hidden_dropout_ratio)
        self.activation_fn = util.get_activation_fn(activation=config.activation_fn)
        self.activation_dropout_module = Dropout(float(config.activation_dropout_ratio))
        self.normalize_before = config.pre_layer_norm
        self.fc1 = QuantLinear(
            self.embed_dim,
            config.intermediate_size,
        )
        self.fc2 = QuantLinear(
            config.intermediate_size,
            self.embed_dim,
            pre_activation=config.activation_fn,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

        if initial_weights is None or initial_biases is None:
            return

        # load initial weights
        self.self_attn.qkv_proj.weight.data.copy_(
            copy_para(torch.cat(initial_weights[:3], 0), config.fp16)
        )
        self.self_attn.qkv_proj.bias.data.copy_(
            copy_para(torch.cat(initial_biases[:3], 0), config.fp16)
        )
        self.self_attn.out_proj.weight.data.copy_(
            copy_para(initial_weights[3], config.fp16)
        )
        self.self_attn.out_proj.bias.data.copy_(
            copy_para(initial_biases[3], config.fp16)
        )
        self.self_attn_layer_norm.weight.data.copy_(
            copy_para(initial_weights[4], config.fp16)
        )
        self.self_attn_layer_norm.bias.data.copy_(
            copy_para(initial_biases[4], config.fp16)
        )
        self.fc1.weight.data.copy_(copy_para(initial_weights[5], config.fp16))
        self.fc1.bias.data.copy_(copy_para(initial_biases[5], config.fp16))
        self.fc2.weight.data.copy_(copy_para(initial_weights[6], config.fp16))
        self.fc2.bias.data.copy_(copy_para(initial_biases[6], config.fp16))
        self.final_layer_norm.weight.data.copy_(
            copy_para(initial_weights[7], config.fp16)
        )
        self.final_layer_norm.bias.data.copy_(copy_para(initial_biases[7], config.fp16))

    def build_self_attention(self, embed_dim, nhead, attn_dropout):
        return MultiheadAttention(
            embed_dim,
            nhead,
            dropout=attn_dropout,
            self_attention=True,
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

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.


        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)`
        """

        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        x = x.transpose(0, 1)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
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

        x = x.transpose(0, 1)

        return x


class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Decoder layer implemented by fairseq.
    This version only removes the "args" parameter, no other changes
    """

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.dropout_module = Dropout(config.hidden_dropout_ratio)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            config.nhead,
            config.attn_prob_dropout_ratio,
            has_causal_mask=not config.has_cross_attn,
        )

        self.activation_fn = util.get_activation_fn(activation=config.activation_fn)
        self.activation_dropout_module = Dropout(float(config.activation_dropout_ratio))
        self.normalize_before = config.pre_layer_norm
        self.has_cross_attn = config.has_cross_attn

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if config.has_cross_attn:
            self.encoder_attn = self.build_encoder_attention(
                self.embed_dim,
                config.hidden_size,
                config.attn_prob_dropout_ratio,
                config.nhead,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = QuantLinear(
            self.embed_dim,
            config.intermediate_size,
        )
        self.fc2 = QuantLinear(
            config.intermediate_size,
            self.embed_dim,
            pre_activation=config.activation_fn,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

        if initial_weights is None or initial_biases is None:
            return

        # load initial weights
        self.self_attn.qkv_proj.weight.data.copy_(
            copy_para(torch.cat(initial_weights[:3], 0), config.fp16)
        )
        self.self_attn.qkv_proj.bias.data.copy_(
            copy_para(torch.cat(initial_biases[:3], 0), config.fp16)
        )
        self.self_attn.out_proj.weight.data.copy_(
            copy_para(initial_weights[3], config.fp16)
        )
        self.self_attn.out_proj.bias.data.copy_(
            copy_para(initial_biases[3], config.fp16)
        )
        self.self_attn_layer_norm.weight.data.copy_(
            copy_para(initial_weights[4], config.fp16)
        )
        self.self_attn_layer_norm.bias.data.copy_(
            copy_para(initial_biases[4], config.fp16)
        )
        if config.has_cross_attn:
            self.encoder_attn.q_proj.weight.data.copy_(
                copy_para(initial_weights[5], config.fp16)
            )
            self.encoder_attn.q_proj.bias.data.copy_(
                copy_para(initial_weights[5], config.fp16)
            )
            self.encoder_attn.k_proj.weight.data.copy_(
                copy_para(initial_weights[6], config.fp16)
            )
            self.encoder_attn.k_proj.bias.data.copy_(
                copy_para(initial_weights[6], config.fp16)
            )
            self.encoder_attn.v_proj.weight.data.copy_(
                copy_para(initial_weights[7], config.fp16)
            )
            self.encoder_attn.v_proj.bias.data.copy_(
                copy_para(initial_weights[7], config.fp16)
            )
            self.encoder_attn.out_proj.weight.data.copy_(
                copy_para(initial_weights[8], config.fp16)
            )
            self.encoder_attn.out_proj.bias.data.copy_(
                copy_para(initial_biases[8], config.fp16)
            )
            self.encoder_attn_layer_norm.weight.data.copy_(
                copy_para(initial_weights[9], config.fp16)
            )
            self.encoder_attn_layer_norm.bias.data.copy_(
                copy_para(initial_biases[9], config.fp16)
            )
            self.fc1.weight.data.copy_(copy_para(initial_weights[10], config.fp16))
            self.fc1.bias.data.copy_(copy_para(initial_biases[10], config.fp16))
            self.fc2.weight.data.copy_(copy_para(initial_weights[11], config.fp16))
            self.fc2.bias.data.copy_(copy_para(initial_biases[11], config.fp16))
            self.final_layer_norm.weight.data.copy_(
                copy_para(initial_weights[12], config.fp16)
            )
            self.final_layer_norm.bias.data.copy_(
                copy_para(initial_biases[12], config.fp16)
            )
        else:
            self.fc1.weight.data.copy_(copy_para(initial_weights[5], config.fp16))
            self.fc1.bias.data.copy_(copy_para(initial_biases[5], config.fp16))
            self.fc2.weight.data.copy_(copy_para(initial_weights[6], config.fp16))
            self.fc2.bias.data.copy_(copy_para(initial_biases[6], config.fp16))
            self.final_layer_norm.weight.data.copy_(
                copy_para(initial_weights[7], config.fp16)
            )
            self.final_layer_norm.bias.data.copy_(
                copy_para(initial_biases[7], config.fp16)
            )

    def build_self_attention(
        self,
        embed_dim,
        nhead,
        attn_dropout,
        add_bias_kv=False,
        add_zero_attn=False,
        has_causal_mask=False,
    ):
        return MultiheadAttention(
            embed_dim,
            nhead,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            is_decoder=True,
            has_causal_mask=has_causal_mask,
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
            is_decoder=True,
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
            x (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
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
        x = x.transpose(0, 1)
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

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.has_cross_attn and encoder_out is not None:
            if (
                encoder_out.shape[1] != x.shape[1]
                and x.shape[1] % encoder_out.shape[1] == 0
            ):
                beam_size = int(x.shape[1] / encoder_out.shape[1])
                encoder_out = encoder_out.repeat_interleave(beam_size, 1)
                encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                    beam_size, 0
                )

            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
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
                x = self.encoder_attn_layer_norm(x)

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
        x = x.transpose(0, 1)
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


class TransformerEmbeddingLayer(TransformerEmbeddingLayerBase):
    def __init__(self, config, initial_embeddings=None, emb_lookup=None):
        super().__init__()

        if emb_lookup is not None:
            self.emb_lookup = emb_lookup
        else:
            self.emb_lookup = nn.Embedding(
                config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
            )
        self.emb_lookup.to(dtype=(torch.half if config.fp16 else torch.float))
        self.embeddings = self.emb_lookup.weight
        nn.init.normal_(self.embeddings, mean=0, std=config.embedding_dim**-0.5)
        nn.init.constant_(self.embeddings[config.padding_idx], 0)

        # load initial weights
        if initial_embeddings is not None:
            self.emb_lookup.weight.data.copy_(
                copy_para(initial_embeddings, config.fp16)
            )

        if config.trainable_pos:
            if config.need_offset:
                num_embeddings = config.max_seq_len + config.padding_idx + 1
            else:
                num_embeddings = config.max_seq_len
            self.embed_positions = TrainablePositionalEmbedding(
                num_embeddings,
                config.embedding_dim,
                config.padding_idx,
                config.need_offset,
            )
            nn.init.normal_(
                self.embed_positions.weight, mean=0, std=config.embedding_dim**-0.5
            )
            if config.need_offset is not None:
                nn.init.constant_(self.embed_positions.weight[config.padding_idx], 0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.embedding_dim,
                config.padding_idx,
                config.max_seq_len,
                config.fp16,
            )
        self.embedding_dim = config.embedding_dim
        self.dropout = Dropout(config.dropout)
        self.emb_quant = TensorQuantizer(emb_quant_config)
        if config.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(config.embedding_dim)
        else:
            self.layernorm_embedding = None
        self.embed_scale = (
            1.0 if config.no_scale_embedding else math.sqrt(config.embedding_dim)
        )
        self.config = config

    def forward(self, input, step=0):
        x = self.emb_lookup(input)
        x = self.emb_quant(x)
        x = self.embed_scale * x
        x += self.embed_positions(input, step)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout(x)

        return x


class TrainablePositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, need_offset):
        super().__init__(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.padding_id = padding_idx
        self.offset = padding_idx + 1 if need_offset else 0

    def make_positions(self, tensor, step):
        mask = tensor.ne(self.padding_id).int()
        positions = torch.cumsum(mask, dim=1).type_as(mask) - 1 + step + self.offset
        return (positions * mask).long()

    def forward(self, input, step: int = 0):
        """`input.shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input.size(0), input.size(1)
        positions = self.make_positions(input, step)
        mask = (
            torch.ne(input, self.padding_id)
            .unsqueeze(2)
            .expand(bsz, seq_len, self.embedding_dim)
        )
        return super().forward(positions) * mask


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
        )
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

    def make_positions(self, tensor, padding_idx, step):
        mask = tensor.ne(padding_idx).int()
        return ((torch.cumsum(mask, dim=1).type_as(mask) - 1 + step) * mask).long()

    def forward(
        self,
        input,
        step=0,
        incremental_state=None,
        timestep=None,
        positions=None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size(0), input.size(1)
        positions = self.make_positions(input, self.padding_idx, step)
        mask = (
            torch.ne(input, self.padding_idx)
            .unsqueeze(2)
            .expand(bsz, seq_len, self.embedding_dim)
        )
        return (
            self.weights.to(input.device)
            .index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            * mask
        ).detach()


class BertEmbeddingLayer(TransformerEmbeddingLayerBase):
    def __init__(self, config, initial_weights=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_len, config.embedding_dim
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_dim
        )

        self.LayerNorm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "position_ids", torch.arange(config.max_seq_len).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        self.emb_quant = TensorQuantizer(emb_quant_config)

        if initial_weights is None:
            return

        # load initial weights
        self.word_embeddings.weight.data.copy_(
            copy_para(initial_weights[0], config.fp16)
        )
        self.position_embeddings.weight.data.copy_(
            copy_para(initial_weights[1], config.fp16)
        )
        self.token_type_embeddings.weight.data.copy_(
            copy_para(initial_weights[2], config.fp16)
        )
        self.LayerNorm.weight.data.copy_(copy_para(initial_weights[3], config.fp16))
        self.LayerNorm.bias.data.copy_(copy_para(initial_weights[4], config.fp16))

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        assert input_ids is not None
        assert position_ids is None
        assert inputs_embeds is None
        # assert torch.all(token_type_ids == 0)

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        token_type_ids = self.token_type_ids[:, :seq_length].expand(
            input_shape[0], seq_length
        )

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.emb_quant(embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
