import math
from dataclasses import dataclass
from tests.fairseq_layers import TransformerEmbeddingLayer
from typing import Dict

import torch
from torch import nn
from torch.autograd import Function


from lightseq.training.ops.pytorch.builder import TransformerBuilder
from lightseq.training.ops.pytorch.util import state_dict, get_pos_embedding


transformer_cuda_module = TransformerBuilder().load()


@dataclass
class LSEmbeddingConfig:
    vocab_size: int  # vocabulary size
    embedding_dim: int  # embedding size
    max_batch_tokens: int  # max batch token numbers
    max_seq_len: int  # max sequence length
    padding_idx: int  # padding token id in vocabulary
    dropout: float  # embedding dropout ration
    fp16: bool  # fp16 presion
    local_rank: int  # rank in local node
    training: bool = True  # training or not


class LSTransformerEmbeddingFunc(Function):
    @staticmethod
    def forward(
        ctx,
        config: LSEmbeddingConfig,
        input: torch.Tensor,
        embeddings: torch.Tensor,
        step: int,
        layer_id: int,
        training: bool,
        is_grad_enabled: bool,
        _all_layer_grads: Dict[int, torch.Tensor],
    ):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.transformer_embedding_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_embedding_layer_fw_fp32
        )

        (output,) = forward_func(layer_id, input, step, training)
        ctx.training = training
        ctx.layer_id = layer_id
        ctx._all_layer_grads = _all_layer_grads

        if is_grad_enabled and training:
            ctx.save_for_backward(input)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cuda_module = transformer_cuda_module
        backward_func = (
            cuda_module.transformer_embedding_layer_bw_fp16
            if ctx.config.fp16
            else cuda_module.transformer_embedding_layer_bw_fp32
        )
        assert ctx.training

        (input,) = ctx.saved_tensors

        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)

        backward_func(ctx.layer_id, grad_output, input)

        grad = ctx._all_layer_grads[ctx.layer_id]

        return (None, None, grad, None, None, None, None, None)


class LSTransformerEmbeddingLayer(nn.Module):
    """Initialize the Lightseq Embedding Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
    Arguments:
        config: An object of LSTransformerEmbeddingLayer config, see get_config

        initial_embeddings: Optional: Only used for unit test
    """

    layer_id: int = 0
    __all_layer_grads: Dict[int, torch.Tensor] = dict()

    # @torch.jit.export
    def __init__(
        self,
        config: LSEmbeddingConfig,
        initial_embeddings=None,
    ):
        # super(LSTransformerEmbeddingLayer, self).__init__()
        super().__init__()

        self.config = config
        self.fp16 = config.fp16
        self._layer_id = LSTransformerEmbeddingLayer.layer_id
        self.max_batch_tokens = config.max_batch_tokens
        self.max_seq_len = config.max_seq_len
        self._all_layer_grads = LSTransformerEmbeddingLayer.__all_layer_grads
        LSTransformerEmbeddingLayer.layer_id = self._layer_id + 1

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        if initial_embeddings is None:
            self.embeddings = nn.Parameter(
                torch.Tensor(self.config.vocab_size, self.config.embedding_dim)
            )
            self.reset_parameters()
        else:
            self.embeddings = torch.nn.Parameter(
                torch.empty_like(initial_embeddings).copy_(initial_embeddings)
            )

        if self.fp16 and self.embeddings.dtype != torch.half:
            self.register_buffer("para_16", self.embeddings.clone().detach().half())

        self.pos_embeddings = get_pos_embedding(
            self.config.max_seq_len, self.config.embedding_dim
        ).to(self.config.local_rank)
        if self.config.fp16:
            self.pos_embeddings = self.pos_embeddings.to(torch.half)

        # Load cuda modules if needed
        # global transformer_cuda_module
        # if transformer_cuda_module is None:
        #     transformer_cuda_module = TransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_transformer_embedding_layer_fp16
            if self.config.fp16
            else cuda_module.create_transformer_embedding_layer_fp32
        )

        create_layer_func(
            self._layer_id,
            self.pos_embeddings,
            config.max_batch_tokens,
            config.embedding_dim,
            config.vocab_size,
            config.dropout,
            config.padding_idx,
        )

    @staticmethod
    def get_config(
        vocab_size: int,
        embedding_dim: int,
        max_batch_tokens: int,
        max_seq_len: int,
        padding_idx: int,
        dropout: float,
        fp16: bool,
        local_rank: int,
    ):

        return LSEmbeddingConfig(
            vocab_size,
            embedding_dim,
            max_batch_tokens,
            max_seq_len,
            padding_idx,
            dropout,
            fp16,
            local_rank,
        )

    def reset_parameters(self):
        nn.init.normal_(self.embeddings, mean=0, std=self.config.embedding_dim ** -0.5)
        nn.init.constant_(self.embeddings[self.config.padding_idx], 0)

    def _assign_layer_weight_grad(self):
        param = (
            self.para_16
            if self.fp16 and self.embeddings.dtype != torch.half
            else self.embeddings
        )

        if self._layer_id in self._all_layer_grads:
            return

        cuda_module = transformer_cuda_module
        grad = torch.empty_like(param)
        if self.fp16:
            # func = cuda_module.assign_layer_weight_grad_fp16
            torch.ops.lightseq_ops.assign_layer_weight_grad_fp16(
                param, grad, "TransformerEmbeddingLayer", self._layer_id
            )
        else:
            torch.ops.lightseq_ops.assign_layer_weight_grad_fp32(
                param, grad, "TransformerEmbeddingLayer", self._layer_id
            )
            # func = cuda_module.assign_layer_weight_grad_fp32
        # func(param, grad, "TransformerEmbeddingLayer", self._layer_id)
        self._all_layer_grads[self._layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(self, input: torch.Tensor, step: int = 0):
        if hasattr(self, "para_16"):
            self.para_16.copy_(self.embeddings.to(torch.half))

        self._assign_layer_weight_grad()
        input = input.to(torch.int)
        bs, sl = input.size()
        if bs * sl > self.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit"
                f" {self.max_batch_tokens}."
            )
        if sl > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.max_seq_len}."
            )
        if step >= self.max_seq_len:
            raise ValueError(
                f"Target sequence length {sl} exceeds the limit {self.max_seq_len}."
            )
        # x = LSTransformerEmbeddingFunc.apply(
        #     self.config,
        #     input,
        #     self.embeddings,
        #     step,
        #     self.layer_id,
        #     self.training,
        #     torch.is_grad_enabled(),
        #     self._all_layer_grads,
        # )
        x = torch.ops.lightseq_ops.transformer_embedding(
            input,
            self.embeddings,
            # self._all_layer_grads[self._layer_id],
            self._layer_id,
            step,
            self.training,
            torch.is_grad_enabled(),
            self.fp16,
        )
        return x.to(self.embeddings)
