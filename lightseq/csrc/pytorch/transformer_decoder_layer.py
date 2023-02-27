import __init__
from itertools import zip_longest
import copy
import torch
from torch import nn
import math
from dataclasses import dataclass

from csrc.pytorch.layer_base import TransformerDecoderLayerBase
from csrc.pytorch.util import (
    copy_para,
    state_dict,
    calc_offset,
)
from csrc.pytorch.torch_transformer_layers import (
    act_quant_config,
    weight_quant_config
)

from csrc.pytorch.builder.cuda_layer_builder import CudaLayerBuilder

cuda_layer_module = CudaLayerBuilder().load()

_all_layer_grads = dict()


class LSTransformerDecoderFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        decoder_states,
        encoder_out,
        encoder_padding_mask,
        config,
        cache,
    ):
        cuda_module = cuda_layer_module
        forward_func = (
            cuda_module.transformer_decoder_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_decoder_layer_fw_fp32
        )

        (output,) = forward_func(
            config.layer_id,
            decoder_states,
            encoder_out,
            encoder_padding_mask,
            config.pre_layer_norm,
            config.quant_mode,
            cache,
            0,
        )

        return output


class LSTransformerDecoderLayer(TransformerDecoderLayerBase):
    """Initialize the Lightseq Transformer Decoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerDecoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super(LSTransformerDecoderLayer, self).__init__()

        self.config = copy.deepcopy(config)
        self.config.layer_id = LSTransformerDecoderLayer.layer_id
        LSTransformerDecoderLayer.layer_id = LSTransformerDecoderLayer.layer_id + 1

        print("Lightseq Transformer config is ", self.config.__dict__)

        self.quant_mode = False

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        # create the layer in cuda kernels.
        cuda_module = cuda_layer_module
        create_layer_func = (
            cuda_module.create_transformer_decoder_layer_new_fp16
            if self.config.fp16
            else cuda_module.create_transformer_decoder_layer_new_fp32
        )

        create_layer_func(
            self.config.nlayer,
            self.config.layer_id,
            self.config.max_batch_tokens,
            self.config.max_seq_len,
            self.config.hidden_size,
            self.config.nhead,
            self.config.intermediate_size,
            self.config.attn_prob_dropout_ratio,
            self.config.activation_dropout_ratio,
            self.config.hidden_dropout_ratio,
            self.config.pre_layer_norm,
            self.config.activation_fn,
        )

        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        self.hs = hs
        self.ims = ims

        self.para_offset = LSTransformerDecoderLayer.gen_offset(
            hs, ims, self.config.nlayer
        )
        if self.config.layer_id != 0:
            self.para_offset = self.para_offset[:-2]
        self.para = torch.nn.Parameter(torch.Tensor(self.para_offset[-1]))

        self.__cache_list = [torch.zeros((self.config.max_batch_tokens, self.config.hidden_size), dtype=torch.half, device="cuda:0") for _ in range(4)]

        # if initial_weights is None or initial_biases is None:
        #     # enc-dec kv weights and bias
        #     self.init_transformer_weights()
        #     return

        # For testing only.
        attn_qkvw = [ele.detach().clone() for ele in initial_weights[:3]]
        attn_qkvw = torch.cat(attn_qkvw, dim=0)
        weights = [attn_qkvw] + [
            copy_para(ele) if ele is not None else None for ele in initial_weights[3:]
        ]

        attn_qkvb = [ele.detach().clone() for ele in initial_biases[:3]]
        attn_qkvb = torch.cat(attn_qkvb, dim=0)
        biases = [attn_qkvb] + [
            copy_para(ele) if ele is not None else None for ele in initial_biases[3:]
        ]

        idx = 0
        for w, b in zip_longest(weights, biases):
            if w is not None:
                cur_para = self._get_weights(idx)
                assert cur_para.numel() == w.numel()
                cur_para.copy_(w.view(-1))
                idx += 1

            if b is not None:
                cur_para = self._get_weights(idx)
                assert cur_para.numel() == b.numel()
                cur_para.copy_(b.view(-1))
                idx += 1

    @staticmethod
    def gen_offset(hidden_size, intermediate_size, nlayer):
        """Returns the offset of each module's parameters among all
        parameters of a layer
        """
        hs, ims = hidden_size, intermediate_size
        sizes = [
            hs * hs * 3,  # attn_qkvw
            hs * 3,  # attn_qkvb
            hs * hs,  # attn_ow
            hs,  # attn_ob
            hs,  # attn_nw
            hs,  # attn_nb
            hs * hs,  # encdec_attn_qw
            hs,  # encdec_attn_qb
            hs * hs,  # encdec_attn_ow
            hs,  # encdec_attn_ob
            hs,  # encdec_attn_nw
            hs,  # encdec_attn_nb
            hs * ims,  # inter_w
            ims,  # inter_b
            hs * ims,  # output_w
            hs,  # output_b
            hs,  # ffn_nw
            hs,  # ffn_nb
            24,
            hs * hs * 2 * nlayer,  # encdec_attn_kvw
            hs * 2 * nlayer,  # encdec_attn_kvb
        ]
        offsets = calc_offset(sizes)
        return offsets

    def params_dict(self):
        """
        Returns:
            weight: dict
            bias: dict
        """

        def copy_and_view(m, shape=None):
            if shape is None:
                shape = (-1,)
            return m.data.clone().view(*shape)

        def _copy(m):
            return copy_and_view(m, (self.hs, self.hs))

        self_attn_qkvw = self._get_weights(0)
        self_attn_qw, self_attn_kw, self_attn_vw = self_attn_qkvw.split(
            self.hs * self.hs, 0
        )
        self_attn_qkvb = self._get_weights(1)
        self_attn_qb, self_attn_kb, self_attn_vb = self_attn_qkvb.split(self.hs, 0)

        all_enc_attn_kw, all_enc_attn_vw = None, None
        all_enc_attn_kb, all_enc_attn_vb = None, None
        if self.config.layer_id == 0:
            all_enc_attn_kvw = self._get_weights(19)
            all_enc_attn_kvw = all_enc_attn_kvw.split(self.hs * self.hs, 0)
            all_enc_attn_kw = list(map(_copy, all_enc_attn_kvw[::2]))
            all_enc_attn_vw = list(map(_copy, all_enc_attn_kvw[1::2]))

            all_enc_attn_kvb = self._get_weights(20)
            all_enc_attn_kvb = all_enc_attn_kvb.split(self.hs, 0)
            all_enc_attn_kb = list(map(copy_and_view, all_enc_attn_kvb[::2]))
            all_enc_attn_vb = list(map(copy_and_view, all_enc_attn_kvb[1::2]))

        weight = {
            "self_attn.q_proj": copy_and_view(self_attn_qw, (self.hs, self.hs)),
            "self_attn.k_proj": copy_and_view(self_attn_kw, (self.hs, self.hs)),
            "self_attn.v_proj": copy_and_view(self_attn_vw, (self.hs, self.hs)),
            "self_attn.out_proj": copy_and_view(
                self._get_weights(2), (self.hs, self.hs)
            ),
            "self_attn_layer_norm": copy_and_view(self._get_weights(4), (self.hs,)),
            "encoder_attn.q_proj": copy_and_view(
                self._get_weights(6), (self.hs, self.hs)
            ),
            "encoder_attn.out_proj": copy_and_view(
                self._get_weights(8), (self.hs, self.hs)
            ),
            "encoder_attn_layer_norm": copy_and_view(self._get_weights(10), (self.hs,)),
            "fc1": copy_and_view(self._get_weights(12), (self.ims, self.hs)),
            "fc2": copy_and_view(self._get_weights(14), (self.hs, self.ims)),
            "final_layer_norm": copy_and_view(self._get_weights(16), (self.hs,)),
            "clip_max": copy_and_view(self._get_weights(18), (24,)),
            "encoder_attn.k_proj": all_enc_attn_kw,
            "encoder_attn.v_proj": all_enc_attn_vw,
        }
        bias = {
            "self_attn.q_proj": copy_and_view(self_attn_qb),
            "self_attn.k_proj": copy_and_view(self_attn_kb),
            "self_attn.v_proj": copy_and_view(self_attn_vb),
            "self_attn.out_proj": copy_and_view(self._get_weights(3)),
            "self_attn_layer_norm": copy_and_view(self._get_weights(5)),
            "encoder_attn.q_proj": copy_and_view(self._get_weights(7), (self.hs,)),
            "encoder_attn.out_proj": copy_and_view(self._get_weights(9), (self.hs,)),
            "encoder_attn_layer_norm": copy_and_view(self._get_weights(11), (self.hs,)),
            "fc1": copy_and_view(self._get_weights(13)),
            "fc2": copy_and_view(self._get_weights(15)),
            "final_layer_norm": copy_and_view(self._get_weights(17)),
            "encoder_attn.k_proj": all_enc_attn_kb,
            "encoder_attn.v_proj": all_enc_attn_vb,
        }
        return weight, bias

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def calc_bound(self, w):
        """Used to initialize parameters"""
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def init_transformer_weights(self):
        """
        0 attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
        6 encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb,
        12 inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
        18 encdec_attn_kvw, encdec_attn_kvb,
        """
        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        attn_qkvw = self._get_weights(0).view(-1, hs)
        nn.init.xavier_uniform_(attn_qkvw, 1.0 / math.sqrt(2.0))
        bound = self.calc_bound(attn_qkvw)
        nn.init.uniform_(self._get_weights(1), -bound, bound)

        encdec_attn_qw = self._get_weights(6).view(hs, hs)
        nn.init.xavier_uniform_(encdec_attn_qw, 1.0 / math.sqrt(2.0))
        bound = self.calc_bound(encdec_attn_qw)
        nn.init.uniform_(self._get_weights(7), -bound, bound)

        nn.init.xavier_uniform_(self._get_weights(2).view(hs, hs), 1.0)
        nn.init.zeros_(self._get_weights(3))
        nn.init.xavier_uniform_(self._get_weights(8).view(hs, hs), 1.0)
        nn.init.zeros_(self._get_weights(9))

        inter_w = self._get_weights(12).view(ims, hs)
        nn.init.kaiming_uniform_(inter_w, math.sqrt(5.0))
        bound = self.calc_bound(inter_w)
        nn.init.uniform_(self._get_weights(13), -bound, bound)

        output_w = self._get_weights(14).view(hs, ims)
        nn.init.kaiming_uniform_(output_w, math.sqrt(5.0))
        bound = self.calc_bound(output_w)
        nn.init.uniform_(self._get_weights(15), -bound, bound)

        nn.init.ones_(self._get_weights(4))
        nn.init.zeros_(self._get_weights(5))
        nn.init.ones_(self._get_weights(10))
        nn.init.zeros_(self._get_weights(11))
        nn.init.ones_(self._get_weights(16))
        nn.init.zeros_(self._get_weights(17))

        act_cmax = act_quant_config.amax.tolist()
        wei_cmax = weight_quant_config.amax.tolist()
        init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 8)
        self._get_weights(18).copy_(init_clip_max)

        if self.config.layer_id == 0:
            encdec_attn_kvw = self._get_weights(19).view(-1, hs)
            nn.init.xavier_uniform_(encdec_attn_kvw, 1.0 / math.sqrt(2.0))
            bound = self.calc_bound(encdec_attn_kvw)
            nn.init.uniform_(self._get_weights(20), -bound, bound)

    def __assign_layer_weight_grad(self):
        """fp16 or fp32"""
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )

        if self.config.layer_id in _all_layer_grads:
            return
        grad = torch.zeros_like(param)
        cuda_module = cuda_layer_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        func(param, grad, "TransformerDecoderLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(
        self, decoder_states, encoder_out, encoder_padding_mask, **kwargs
    ):
        """
        decoder_states, [batch_size, trg_len, hidden_size] or [batch_size * beam_size, 1, hidden_size]
        encoder_out, [src_len, batch_size, hidden_size]
        encoder_padding_mask, [batch_size, src_len], 0 for non-pad, 1 for padding
        cache, dict, {"dec_self_k": [batch*beam, nh, step, hd],
                      "dec_self_v": [batch*beam, nh, step, hd],
                      "encdec_kv": [n_dec_layer * 2, batch_size, nhead, src_seq_len, head_dim]
                      }
        """
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        self.config.quant_mode = self.quant_mode

        decoder_states = decoder_states.contiguous()
        # [s, b, h] -> [b, s, h]
        encoder_out = encoder_out.transpose(0, 1).contiguous()
        encoder_padding_mask = (
            (encoder_padding_mask * -1e8).type_as(decoder_states).contiguous()
        )

        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        if self.config.fp16:
            decoder_states = decoder_states.to(torch.half)
            encoder_out = encoder_out.to(torch.half)
            encoder_padding_mask = encoder_padding_mask.to(torch.half)

        self.__assign_layer_weight_grad()
        
        bs, sl, dim = decoder_states.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit"
                f" {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        if len(encoder_padding_mask.size()) == 1:
            assert encoder_out.size(0) == 1 and encoder_out.size(
                1
            ) == encoder_padding_mask.size(0)
        else:
            assert encoder_out.size(0) == encoder_padding_mask.size(
                0
            ) and encoder_out.size(1) == encoder_padding_mask.size(1)
        # if cache is None:
        #     assert bs == encoder_out.size(0)
        # else:
        #     assert bs % encoder_out.size(0) == 0
        output = LSTransformerDecoderFunc.apply(
            decoder_states,
            encoder_out,
            encoder_padding_mask,
            self.config,
            self.__cache_list,
        )
        return output.to(self.para)

    def disable_quant(self):
        self.quant_mode = False

    def enable_quant(self):
        self.quant_mode = True
