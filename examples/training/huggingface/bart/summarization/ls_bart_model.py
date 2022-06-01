import torch
from torch import nn
from dataclasses import dataclass
from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer as TransformerEncoderLayer,
)
from lightseq.training.ops.pytorch.transformer_decoder_layer import (
    LSTransformerDecoderLayer as TransformerDecoderLayer,
)
from transformers import (
    BartForConditionalGeneration,
    BartPretrainedModel,
)
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer


def get_weight_and_bias(m):
    weight = m.weight.detach().clone()
    bias = m.bias.detach().clone()
    return weight, bias


def get_hf_bart_dec_enc_atten_kv(layers, params_list, nlayer):
    init_ws, init_bs = [], []
    for i in range(nlayer):
        layer = layers[i]
        modules_list = []
        exec(f"modules_list.append(layer.{params_list.encoder_attn_k_proj})")
        exec(f"modules_list.append(layer.{params_list.encoder_attn_v_proj})")
        for module in modules_list:
            w, b = get_weight_and_bias(module)
            init_ws.append(w)
            init_bs.append(b)
    enc_attn_kvw = torch.cat([ele for ele in init_ws], dim=0)
    enc_attn_kvb = torch.cat([ele for ele in init_bs], dim=0)
    return enc_attn_kvw, enc_attn_kvb


def get_enc_layer_config(training_args, config):
    enc_config = LSHFTransformerEncoderLayer.get_config(
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.d_model,
        intermediate_size=config.encoder_ffn_dim,
        nhead=config.encoder_attention_heads,
        attn_prob_dropout_ratio=config.attention_dropout,
        activation_dropout_ratio=config.activation_dropout,
        hidden_dropout_ratio=config.dropout,
        activation_fn=config.activation_function,
        max_batch_tokens=4096,
        pre_layer_norm=False,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
    )
    enc_params_list = LSHFTransformerEncoderLayer.get_params_list(
        self_attn_q_proj="self_attn.q_proj",
        self_attn_k_proj="self_attn.k_proj",
        self_attn_v_proj="self_attn.v_proj",
        self_attn_out_proj="self_attn.out_proj",
        self_attn_layer_norm="self_attn_layer_norm",
        fc1="fc1",
        fc2="fc2",
        final_layer_norm="final_layer_norm",
    )
    return enc_config, enc_params_list


def get_dec_layer_config(training_args, config):
    dec_config = LSHFTransformerDecoderLayer.get_config(
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.d_model,
        intermediate_size=config.decoder_ffn_dim,
        nhead=config.decoder_attention_heads,
        attn_prob_dropout_ratio=config.attention_dropout,
        activation_dropout_ratio=config.activation_dropout,
        hidden_dropout_ratio=config.dropout,
        activation_fn=config.activation_function,
        nlayer=config.decoder_layers,
        pre_layer_norm=False,
        max_batch_tokens=4096,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
    )
    dec_params_list = LSHFTransformerDecoderLayer.get_params_list(
        self_attn_q_proj="self_attn.q_proj",
        self_attn_k_proj="self_attn.k_proj",
        self_attn_v_proj="self_attn.v_proj",
        self_attn_out_proj="self_attn.out_proj",
        self_attn_layer_norm="self_attn_layer_norm",
        encoder_attn_q_proj="encoder_attn.q_proj",
        encoder_attn_out_proj="encoder_attn.out_proj",
        encoder_attn_layer_norm="encoder_attn_layer_norm",
        fc1="fc1",
        fc2="fc2",
        final_layer_norm="final_layer_norm",
        encoder_attn_k_proj="encoder_attn.k_proj",
        encoder_attn_v_proj="encoder_attn.v_proj",
    )
    return dec_config, dec_params_list


def inject_lightseq_layer(model, training_args, config):
    # encoder op replace
    model = model.model
    for layer_id in range(config.encoder_layers):
        enc_config, enc_params_list = get_enc_layer_config(training_args, config)
        model.encoder.layers[layer_id] = LSHFTransformerEncoderLayer.build_model(
            enc_config, enc_params_list, model.encoder.layers, layer_id
        ).cuda()
    # decoder op replace
    for layer_id in range(config.decoder_layers):
        dec_config, dec_params_list = get_dec_layer_config(training_args, config)
        model.decoder.layers[layer_id] = LSHFTransformerDecoderLayer.build_model(
            dec_config, dec_params_list, model.decoder.layers, layer_id
        ).cuda()


def hf_state_dict(model):
    """
    Args:
        model: huggingface model replaced with lightseq layer
    Returns:
        Dict: The huggingface state dict
    """

    def unwrap_model(model):
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model

    def inject_hf_layer(config, hf_layer, ls_layer, is_decoder=False):
        if not is_decoder:
            for layer_id in range(config.encoder_layers):
                weight, bias = ls_layer[layer_id].params_dict()
                layer = hf_layer[layer_id]
                layer.self_attn.q_proj.weight.data.copy_(weight["self_attn_q_proj"])
                layer.self_attn.q_proj.bias.data.copy_(bias["self_attn_q_proj"])
                layer.self_attn.k_proj.weight.data.copy_(weight["self_attn_k_proj"])
                layer.self_attn.k_proj.bias.data.copy_(bias["self_attn_k_proj"])
                layer.self_attn.v_proj.weight.data.copy_(weight["self_attn_v_proj"])
                layer.self_attn.v_proj.bias.data.copy_(bias["self_attn_v_proj"])
                layer.self_attn.out_proj.weight.data.copy_(weight["self_attn_out_proj"])
                layer.self_attn.out_proj.bias.data.copy_(bias["self_attn_out_proj"])
                layer.self_attn_layer_norm.weight.data.copy_(
                    weight["self_attn_layer_norm"]
                )
                layer.self_attn_layer_norm.bias.data.copy_(bias["self_attn_layer_norm"])
                layer.fc1.weight.data.copy_(weight["fc1"])
                layer.fc1.bias.data.copy_(bias["fc1"])
                layer.fc2.weight.data.copy_(weight["fc2"])
                layer.fc2.bias.data.copy_(bias["fc2"])
                layer.final_layer_norm.weight.data.copy_(weight["final_layer_norm"])
                layer.final_layer_norm.bias.data.copy_(bias["final_layer_norm"])
        else:
            encoder_attn_k_proj_w = None
            encoder_attn_k_proj_b = None
            encoder_attn_v_proj_w = None
            encoder_attn_v_proj_b = None
            for layer_id in range(config.decoder_layers):
                weight, bias = ls_layer[layer_id].params_dict()
                layer = hf_layer[layer_id]
                layer.self_attn.q_proj.weight.data.copy_(weight["self_attn_q_proj"])
                layer.self_attn.q_proj.bias.data.copy_(bias["self_attn_q_proj"])
                layer.self_attn.k_proj.weight.data.copy_(weight["self_attn_k_proj"])
                layer.self_attn.k_proj.bias.data.copy_(bias["self_attn_k_proj"])
                layer.self_attn.v_proj.weight.data.copy_(weight["self_attn_v_proj"])
                layer.self_attn.v_proj.bias.data.copy_(bias["self_attn_v_proj"])
                layer.self_attn.out_proj.weight.data.copy_(weight["self_attn_out_proj"])
                layer.self_attn.out_proj.bias.data.copy_(bias["self_attn_out_proj"])
                layer.self_attn_layer_norm.weight.data.copy_(
                    weight["self_attn_layer_norm"]
                )
                layer.self_attn_layer_norm.bias.data.copy_(bias["self_attn_layer_norm"])
                layer.fc1.weight.data.copy_(weight["fc1"])
                layer.fc1.bias.data.copy_(bias["fc1"])
                layer.fc2.weight.data.copy_(weight["fc2"])
                layer.fc2.bias.data.copy_(bias["fc2"])
                layer.final_layer_norm.weight.data.copy_(weight["final_layer_norm"])
                layer.final_layer_norm.bias.data.copy_(bias["final_layer_norm"])

                layer.encoder_attn.q_proj.weight.data.copy_(
                    weight["encoder_attn_q_proj"]
                )
                layer.encoder_attn.q_proj.bias.data.copy_(bias["encoder_attn_q_proj"])
                layer.encoder_attn.out_proj.weight.data.copy_(
                    weight["encoder_attn_out_proj"]
                )
                layer.encoder_attn.out_proj.bias.data.copy_(
                    bias["encoder_attn_out_proj"]
                )
                layer.encoder_attn_layer_norm.weight.data.copy_(
                    weight["encoder_attn_layer_norm"]
                )
                layer.encoder_attn_layer_norm.bias.data.copy_(
                    bias["encoder_attn_layer_norm"]
                )
                if layer_id == 0:
                    encoder_attn_k_proj_w = weight["encoder_attn_k_proj"]
                    encoder_attn_k_proj_b = bias["encoder_attn_k_proj"]
                    encoder_attn_v_proj_w = weight["encoder_attn_v_proj"]
                    encoder_attn_v_proj_b = bias["encoder_attn_v_proj"]
                layer.encoder_attn.k_proj.weight.data.copy_(
                    encoder_attn_k_proj_w[layer_id]
                )
                layer.encoder_attn.k_proj.bias.data.copy_(
                    encoder_attn_k_proj_b[layer_id]
                )
                layer.encoder_attn.v_proj.weight.data.copy_(
                    encoder_attn_v_proj_w[layer_id]
                )
                layer.encoder_attn.v_proj.bias.data.copy_(
                    encoder_attn_v_proj_b[layer_id]
                )

    model_to_save = unwrap_model(model)
    if not isinstance(model_to_save, LSBartPretrainedModel):
        raise ValueError("Must be ligtseq replaced model")
    # reload original modules
    ls_encoder_layer = model_to_save.model.encoder.layers
    ls_decoder_layer = model_to_save.model.decoder.layers
    model_to_save.model.encoder.layers = nn.ModuleList(
        [BartEncoderLayer(model.config) for _ in range(model.config.encoder_layers)]
    )
    model_to_save.model.decoder.layers = nn.ModuleList(
        [BartDecoderLayer(model.config) for _ in range(model.config.decoder_layers)]
    )

    inject_hf_layer(model.config, model_to_save.model.encoder.layers, ls_encoder_layer)
    inject_hf_layer(
        model.config,
        model_to_save.model.decoder.layers,
        ls_decoder_layer,
        is_decoder=True,
    )
    state_dict = model_to_save.state_dict()
    # replace with lightseq modules
    model_to_save.model.encoder.layers = ls_encoder_layer
    model_to_save.model.decoder.layers = ls_decoder_layer
    return state_dict


class LSHFTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        self.params_list = None
        super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
        ls_encoder_padding_mask = (
            encoder_padding_mask.narrow(2, 0, 1)
            .squeeze()
            .ne(0)
            .type_as(encoder_padding_mask)
        )
        output = super().forward(hidden_states, ls_encoder_padding_mask)
        return (output, None, None, None)

    @staticmethod
    def get_params_list(**kwargs):
        """Configuration of model hyperparameters for encoder and decoder"""

        @dataclass
        class ParamsList:
            self_attn_q_proj: None
            self_attn_k_proj: None
            self_attn_v_proj: None
            self_attn_out_proj: None
            self_attn_layer_norm: None
            fc1: None
            fc2: None
            final_layer_norm: None

        params_list = ParamsList(**kwargs)
        # check_config(config)
        return params_list

    @classmethod
    def build_model(cls, config, params_list, layer_list, layer_id):
        layer = layer_list[layer_id]
        modules_list = []
        # only python >= 3.6 (orderedDict)
        for module_name in params_list.__dict__.values():
            exec(f"modules_list.append(layer.{module_name})")
        init_ws = []
        init_bs = []
        for module in modules_list:
            w, b = get_weight_and_bias(module)
            init_ws.append(w)
            init_bs.append(b)
        return cls(config, init_ws, init_bs)


class LSHFTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFTransformerDecoderLayer, self).__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=False,
        *args,
        **kwargs,
    ):
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
        ls_encoder_padding_mask = (
            encoder_attention_mask.narrow(2, 0, 1)
            .squeeze()
            .ne(0)
            .type_as(encoder_attention_mask)
        )
        cache = None
        if use_cache:
            cache = (
                {}
                if past_key_value is None
                else {"dec_self_k": past_key_value[0], "dec_self_v": past_key_value[1]}
            )
        output = super().forward(
            hidden_states, encoder_hidden_states, ls_encoder_padding_mask, cache
        )
        return output, (cache["dec_self_k"], cache["dec_self_v"])

    @staticmethod
    def get_params_list(**kwargs):
        """Configuration of model hyperparameters for encoder and decoder"""

        @dataclass
        class ParamsList:
            self_attn_q_proj: None
            self_attn_k_proj: None
            self_attn_v_proj: None
            self_attn_out_proj: None
            self_attn_layer_norm: None
            encoder_attn_q_proj: None
            encoder_attn_out_proj: None
            encoder_attn_layer_norm: None
            fc1: None
            fc2: None
            final_layer_norm: None
            encoder_attn_k_proj: None
            encoder_attn_v_proj: None

        params_list = ParamsList(**kwargs)
        # check_config(config)
        return params_list

    @classmethod
    def build_model(cls, config, params_list, layer_list, layer_id):
        layer = layer_list[layer_id]
        modules_list = []
        for param_name in list(params_list.__dict__.values())[:-2]:
            exec(f"modules_list.append(layer.{param_name})")

        init_ws = []
        init_bs = []
        for module in modules_list:
            w, b = get_weight_and_bias(module)
            init_ws.append(w)
            init_bs.append(b)
        if layer_id == 0:
            enc_kvw, enc_kvb = get_hf_bart_dec_enc_atten_kv(
                layer_list, params_list, config.nlayer
            )
            init_ws.append(enc_kvw)
            init_bs.append(enc_kvb)
        return cls(config, init_ws, init_bs)


class LSBartPretrainedModel(BartPretrainedModel):
    @classmethod
    def from_pretrained(self, *args, training_args, **kwargs):
        self.config = kwargs["config"]
        model = super().from_pretrained(*args, **kwargs)
        inject_lightseq_layer(model, training_args, self.config)
        return model

    def save_pretrained(self, *args, **kwargs):
        kwargs["state_dict"] = hf_state_dict(self)
        super().save_pretrained(*args, **kwargs)


class LSBartForConditionalGeneration(
    LSBartPretrainedModel, BartForConditionalGeneration
):
    """from BartForConditionalGeneration"""


class LSBartForSequenceClassification(
    LSBartPretrainedModel, BartForSequenceClassification
):
    """from BartForSequenceClassification"""


class LSBartForQuestionAnswering(LSBartPretrainedModel, BartForQuestionAnswering):
    """from BartForQuestionAnswering"""
