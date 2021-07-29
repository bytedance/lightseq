import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


from lightseq.training.ops.pytorch.builder import TransformerBuilder
from lightseq.training.ops.pytorch.util import state_dict


transformer_cuda_module = None
_all_layer_grads = dict()


class LSTransformerEmbeddingFunc(Function):
    @staticmethod
    def forward(ctx, config, input, embeddings, step):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.transformer_embedding_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_embedding_layer_fw_fp32
        )

        (output,) = forward_func(config.layer_id, input, step, config.training)

        if config.is_grad_enabled and config.training:
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
        assert ctx.config.training

        (input,) = ctx.saved_tensors

        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)

        backward_func(ctx.config.layer_id, grad_output, input)

        grad = _all_layer_grads[ctx.config.layer_id]

        return (None, None, grad, None)


class LSTransformerEmbeddingLayer(nn.Module):
    """Initialize the Lightseq Embedding Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
    Arguments:
        config: An object of LSTransformerEmbeddingLayer config, see get_config

        initial_embeddings: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(
        self,
        config,
        initial_embeddings=None,
    ):
        super(LSTransformerEmbeddingLayer, self).__init__()

        self.config = config
        self.config.layer_id = LSTransformerEmbeddingLayer.layer_id
        LSTransformerEmbeddingLayer.layer_id = LSTransformerEmbeddingLayer.layer_id + 1

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
        self.pos_embeddings = self.get_pos_embedding(self.config.max_seq_len).to(
            self.config.local_rank
        )
        if self.config.fp16:
            self.pos_embeddings = self.pos_embeddings.to(torch.half)

        # Load cuda modules if needed
        global transformer_cuda_module
        if transformer_cuda_module is None:
            transformer_cuda_module = TransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_transformer_embedding_layer_fp16
            if self.config.fp16
            else cuda_module.create_transformer_embedding_layer_fp32
        )

        create_layer_func(
            self.config.layer_id,
            self.pos_embeddings,
            self.config.max_batch_tokens,
            self.config.embedding_dim,
            self.config.vocab_size,
            self.config.dropout,
            self.config.padding_idx,
        )

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            vocab_size: int  # vocabulary size
            embedding_dim: int  # embedding size
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            padding_idx: int  # padding token id in vocabulary
            dropout: float  # embedding dropout ration
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node

        return Config(**kwargs)

    def reset_parameters(self):
        nn.init.normal_(self.embeddings, mean=0, std=self.config.embedding_dim ** -0.5)
        nn.init.constant_(self.embeddings[self.config.padding_idx], 0)

    def get_pos_embedding(self, num_pos_embeddings):
        half_dim = self.config.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_pos_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_pos_embeddings, -1
        )
        if self.config.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_pos_embeddings, 1)], dim=1)
        return emb

    def __assign_layer_weight_grad(self):
        param = (
            self.para_16
            if self.config.fp16 and self.embeddings.dtype != torch.half
            else self.embeddings
        )

        if self.config.layer_id in _all_layer_grads:
            return

        cuda_module = transformer_cuda_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.empty_like(param)
        func(param, grad, "TransformerEmbeddingLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(self, input, step=0, **kwargs):
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()

        if self.config.fp16 and self.embeddings.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.embeddings.to(torch.half))
            else:
                self.register_buffer("para_16", self.embeddings.clone().detach().half())

        self.__assign_layer_weight_grad()
        input = input.to(torch.int)
        bs, sl = input.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        if step >= self.config.max_seq_len:
            raise ValueError(
                f"Target sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        x = LSTransformerEmbeddingFunc.apply(self.config, input, self.embeddings, step)
        return x.to(self.embeddings)
