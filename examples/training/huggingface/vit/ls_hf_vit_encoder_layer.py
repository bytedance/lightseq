import torch
from lightseq.training.ops.pytorch.quantization import (
    enable_quant,
    disable_quant,
    weight_quant_config,
    act_quant_config,
)


def inject_ls_enc_layer(model, training_args, model_args, config):
    if model_args.module_type == 2:
        from lightseq.training.ops.pytorch.torch_transformer_layers import (
            TransformerEncoderLayer,
        )
    elif model_args.module_type == 1:
        from lightseq.training.ops.pytorch.transformer_encoder_layer import (
            LSTransformerEncoderLayer as TransformerEncoderLayer,
        )
    else:
        raise NotImplementedError

    class LSVITTransformerEncoderLayer(TransformerEncoderLayer):
        def __init__(self, *args, **kwargs):
            super(LSVITTransformerEncoderLayer, self).__init__(*args, **kwargs)

        def forward(self, hidden_states, *args, **kwargs):
            ls_encoder_padding_mask = torch.zeros(hidden_states.size()[:-1]).to(
                hidden_states.device
            )
            output = super().forward(hidden_states, ls_encoder_padding_mask)
            return (output,)

    def gen_vit_config(training_args, config):
        num_patches = (config.image_size // config.patch_size) ** 2 + 1
        max_batch_size = max(
            training_args.per_device_train_batch_size,
            training_args.per_device_eval_batch_size,
        )
        vit_config = TransformerEncoderLayer.get_config(
            max_batch_tokens=num_patches * max_batch_size,
            max_seq_len=num_patches,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.num_attention_heads,
            attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
            activation_dropout_ratio=config.hidden_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            pre_layer_norm=True,
            fp16=training_args.fp16,
            local_rank=training_args.local_rank,
            activation_fn="gelu",
        )
        return vit_config

    for i in range(config.num_hidden_layers):
        vit_config = gen_vit_config(training_args, config)
        init_ws, init_bs = get_hf_vit_enc_layer_params(model.vit.encoder.layer[i])
        model.vit.encoder.layer[i] = LSVITTransformerEncoderLayer(
            vit_config, init_ws, init_bs
        ).cuda()
        if model_args.enable_quant:
            model.vit.encoder.layer[i].apply(enable_quant)
        else:
            model.vit.encoder.layer[i].apply(disable_quant)


def get_hf_vit_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.attention.attention.query.weight.detach().clone())
    init_bs.append(layer.attention.attention.query.bias.detach().clone())
    init_ws.append(layer.attention.attention.key.weight.detach().clone())
    init_bs.append(layer.attention.attention.key.bias.detach().clone())
    init_ws.append(layer.attention.attention.value.weight.detach().clone())
    init_bs.append(layer.attention.attention.value.bias.detach().clone())
    init_ws.append(layer.attention.output.dense.weight.detach().clone())
    init_bs.append(layer.attention.output.dense.bias.detach().clone())
    init_ws.append(layer.layernorm_before.weight.detach().clone())
    init_bs.append(layer.layernorm_before.bias.detach().clone())

    init_ws.append(layer.intermediate.dense.weight.detach().clone())
    init_bs.append(layer.intermediate.dense.bias.detach().clone())
    init_ws.append(layer.output.dense.weight.detach().clone())
    init_bs.append(layer.output.dense.bias.detach().clone())
    init_ws.append(layer.layernorm_after.weight.detach().clone())
    init_bs.append(layer.layernorm_after.bias.detach().clone())

    act_cmax = act_quant_config.amax.tolist()
    wei_cmax = weight_quant_config.amax.tolist()
    init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 4)
    init_ws.append(init_clip_max)

    return init_ws, init_bs
