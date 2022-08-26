import torch

from lightseq.training.ops.pytorch import TransformerBuilder
from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer,
)
from lightseq.training.ops.pytorch.quantization import (
    weight_quant_config,
    act_quant_config,
)

transformer_cuda_module = TransformerBuilder().load()


class LSGptEncoderLayer(LSTransformerEncoderLayer):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super(LSGptEncoderLayer, self).__init__(
            config, initial_weights=initial_weights, initial_biases=initial_biases
        )

    def create_cpp_layer(self):

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_transformer_encoder_layer_fp16
            if self.config.fp16
            else cuda_module.create_transformer_encoder_layer_fp32
        )

        print("create gpt encoder layer")

        create_layer_func(
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
            True,  # mask_future_tokens
        )

    @staticmethod
    def from_huggingface(layer, training_args, model_config):
        ls_gpt_config = gen_ls_gpt_enc_config(training_args, model_config)
        init_ws, init_bs = get_hf_gpt_enc_layer_params(layer, ls_gpt_config)
        return LSHFGptEncoderLayer(ls_gpt_config, init_ws, init_bs).cuda()


class LSHFGptEncoderLayer(LSGptEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFGptEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        # attention mask from transformers is a tensor.
        # sizes are[batch_size, 1, 1, to_seq_length]
        # masked value is -10000.0, unmasked value is 0.0
        if attention_mask is not None:
            ls_attention_mask = attention_mask.squeeze()
        else:
            ls_attention_mask = torch.zeros(hidden_states.size()[:2])
        output = super().forward(hidden_states, ls_attention_mask)
        return (output, None, None, None)


def gen_ls_gpt_enc_config(training_args, config):
    gpt_config = LSGptEncoderLayer.get_config(
        max_batch_tokens=8192,
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        intermediate_size=4 * config.hidden_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attn_pdrop,
        activation_dropout_ratio=config.resid_pdrop,
        hidden_dropout_ratio=config.resid_pdrop,
        pre_layer_norm=True,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
        activation_fn="gelu",
    )
    return gpt_config


def get_hf_gpt_enc_layer_params(layer, gpt_config):
    init_ws = []
    init_bs = []

    init_ws.extend(
        layer.attn.c_attn.weight.detach().clone().t().split(gpt_config.hidden_size, 0)
    )
    init_bs.extend(
        layer.attn.c_attn.bias.detach().clone().split(gpt_config.hidden_size, 0)
    )

    init_ws.append(layer.attn.c_proj.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.attn.c_proj.bias.detach().clone())
    init_ws.append(layer.ln_1.weight.detach().clone())
    init_bs.append(layer.ln_1.bias.detach().clone())

    init_ws.append(layer.mlp.c_fc.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.mlp.c_fc.bias.detach().clone())
    init_ws.append(layer.mlp.c_proj.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.mlp.c_proj.bias.detach().clone())
    init_ws.append(layer.ln_2.weight.detach().clone())
    init_bs.append(layer.ln_2.bias.detach().clone())

    act_cmax = act_quant_config.amax.tolist()
    wei_cmax = weight_quant_config.amax.tolist()
    init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 4)
    init_ws.append(init_clip_max)

    return init_ws, init_bs


def ls_hf_gpt_enc_convert(model, training_args, config):
    for i in range(config.num_hidden_layers):
        model.transformer.h[i] = LSHFGptEncoderLayer.from_huggingface(
            model.transformer.h[i], training_args, config
        ).cuda()
