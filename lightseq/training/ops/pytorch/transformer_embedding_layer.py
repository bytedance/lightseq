import torch
from torch import nn
from torch.autograd import Function

from lightseq.training.ops.pytorch import TransformerBuilder
from lightseq.training.ops.pytorch.layer_base import TransformerEmbeddingLayerBase
from lightseq.training.ops.pytorch.util import (
    copy_para,
    state_dict,
    calc_offset,
    get_pos_embedding,
)

transformer_cuda_module = TransformerBuilder().load()

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

        (output,) = forward_func(
            config.layer_id, input, step, config.training, config.quant_mode
        )

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


class LSTransformerEmbeddingLayer(TransformerEmbeddingLayerBase):
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
        initial_positions=None,
    ):
        super(LSTransformerEmbeddingLayer, self).__init__()

        self.config = config
        self.config.layer_id = LSTransformerEmbeddingLayer.layer_id
        LSTransformerEmbeddingLayer.layer_id = LSTransformerEmbeddingLayer.layer_id + 1

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.quant_mode = False

        self.para_offset = LSTransformerEmbeddingLayer.gen_offset(
            config.embedding_dim, config.vocab_size, config.max_seq_len
        )
        if not config.trainable_pos:
            # only retain the embedding params
            self.para_offset = self.para_offset[:2]

        # if trainable_pos is True, deprecate self.pos_embeddings
        self.pos_embeddings = get_pos_embedding(
            self.config.max_seq_len, self.config.embedding_dim
        ).to(self.config.local_rank)
        if self.config.fp16:
            self.pos_embeddings = self.pos_embeddings.to(torch.half)

        # create the layer in cuda kernels.
        self.create_cpp_layer()

        # declare trainable embedding and position(if needed) parameters
        self.para = nn.Parameter(torch.Tensor(self.para_offset[-1] + 1))
        if initial_embeddings is None and initial_positions is None:
            self.reset_parameters()
            return

        embeddings = self._get_weights(0)
        assert embeddings.numel() == initial_embeddings.numel()
        embeddings.copy_(initial_embeddings.view(-1))
        if config.trainable_pos:
            pos_embeddings = self._get_weights(1)
            assert pos_embeddings.numel() == initial_positions.numel()
            pos_embeddings.copy_(initial_positions.view(-1))

    @property
    def embeddings(self):
        """Returns the embedding parameter without position"""
        return nn.Parameter(
            self._get_weights(0).view(self.config.vocab_size, self.config.embedding_dim)
        )

    def create_cpp_layer(self):
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
            self.config.max_seq_len,
            self.config.dropout,
            self.config.padding_idx,
            self.config.trainable_pos,
        )

    def reset_parameters(self):
        nn.init.normal_(self.para, mean=0, std=self.config.embedding_dim**-0.5)
        embeddings = self._get_weights(0).view(-1, self.config.embedding_dim)
        nn.init.constant_(embeddings[self.config.padding_idx], 0)
        # clip_max
        nn.init.constant_(self.para[-1], 1)

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def __assign_layer_weight_grad(self):
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )

        if self.config.layer_id in _all_layer_grads:
            return

        cuda_module = transformer_cuda_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.zeros_like(param)
        func(param, grad, "TransformerEmbeddingLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    @staticmethod
    def gen_offset(hidden_size, vocab_size, max_position):
        hs, vs, mp = hidden_size, vocab_size, max_position
        sizes = [
            vs * hs,
            mp * hs,
        ]
        offsets = calc_offset(sizes)
        return offsets

    def forward(self, input, step=0, **kwargs):
        self.config.training = self.training
        self.config.quant_mode = self.quant_mode
        self.config.is_grad_enabled = torch.is_grad_enabled()

        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        self.__assign_layer_weight_grad()
        input = input.to(torch.int)
        bs, sl = input.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit"
                f" {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        if step >= self.config.max_seq_len:
            raise ValueError(
                f"Target sequence length {sl} exceeds the limit"
                f" {self.config.max_seq_len}."
            )
        x = LSTransformerEmbeddingFunc.apply(self.config, input, self.para, step)
        return x.to(self.para)

    def disable_quant(self):
        self.quant_mode = False

    def enable_quant(self):
        self.quant_mode = True
