from torch import nn

from lightseq.training.ops.pytorch.quantization import (
    qat_mode,
    disable_quant,
    QuantLinear,
    TensorQuantizer,
    weight_quant_config,
)
from lightseq.training.ops.pytorch.torch_transformer_layers import (
    TransformerDecoderLayer,
    copy_para,
)


def get_hf_gpt_enc_layer_params(layer, config):
    init_ws = []
    init_bs = []

    init_ws.extend(
        layer.attn.c_attn.weight.detach().clone().t().split(config.hidden_size, 0)
    )
    init_bs.extend(layer.attn.c_attn.bias.detach().clone().split(config.hidden_size, 0))

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

    return init_ws, init_bs


def get_hf_gpt_emb_layer_params(layer):
    init_ws = []

    init_ws.append(layer.wte.weight.detach().clone())
    init_ws.append(layer.wpe.weight.detach().clone())

    return init_ws


def gen_gpt_enc_config(training_args, config):
    gpt_enc_config = TransformerDecoderLayer.get_config(
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
        nlayer=config.num_hidden_layers,
        activation_fn="gelu",
        has_cross_attn=False,
    )
    return gpt_enc_config


class LSHFGptEncoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFGptEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        if attention_mask is not None:
            ls_attention_mask = attention_mask.squeeze()
        else:
            ls_attention_mask = torch.zeros(hidden_states.size()[:2])
        output = super().forward(hidden_states, ls_attention_mask)
        return output


class GptEmbedding(nn.Embedding):
    def __init__(self, training_args, initial_embeddings=None, *args, **kwargs):
        super(GptEmbedding, self).__init__(*args, **kwargs)
        self.emb_quant = TensorQuantizer(weight_quant_config)

        if initial_embeddings is not None:
            self.weight.data.copy_(copy_para(initial_embeddings, training_args.fp16))

    def forward(self, input_ids):
        x = super(GptEmbedding, self).forward(input_ids)
        x = self.emb_quant(x)
        return x


def inject_ls_layer(model, training_args, model_args, config):
    if model_args.module_type == 1:
        from lightseq.training import ls_hf_gpt_enc_convert

        ls_hf_gpt_enc_convert(model, training_args, config)
        return

    if model_args.module_type != 2:
        raise NotImplementedError

    init_ws = get_hf_gpt_emb_layer_params(model.transformer)
    model.transformer.wte = GptEmbedding(
        training_args, init_ws[0], config.vocab_size, config.hidden_size
    )
    if model_args.enable_quant:
        model.transformer.wte.apply(qat_mode)
    else:
        model.transformer.wte.apply(disable_quant)

    for i in range(config.num_hidden_layers):
        gpt_enc_config = gen_gpt_enc_config(training_args, config)
        init_ws, init_bs = get_hf_gpt_enc_layer_params(model.transformer.h[i], config)
        model.transformer.h[i] = LSHFGptEncoderLayer(
            gpt_enc_config, init_ws, init_bs
        ).cuda()
        if model_args.enable_quant:
            model.transformer.h[i].apply(qat_mode)
        else:
            model.transformer.h[i].apply(disable_quant)

    q_lm_head = QuantLinear(config.n_embd, config.vocab_size, bias=False)
    q_lm_head.weight = model.transformer.wte.weight
    q_lm_head.weight_quant = model.transformer.wte.emb_quant
    model.lm_head = q_lm_head
