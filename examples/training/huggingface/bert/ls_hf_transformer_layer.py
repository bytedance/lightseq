import torch
from lightseq.training.pytorch_quantization.nn.modules.tensor_quantizer import (
    enable_quant,
)
from lightseq.training.ops.pytorch.quantization import (
    qat_mode,
    disable_quant,
    weight_quant_config,
    act_quant_config,
    relu_quant_config,
)
from lightseq.training.ops.pytorch.torch_transformer_layers import BertEmbeddingLayer
from transformers import (
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertLayer,
    BertLMHeadModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
)


def get_hf_bert_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.attention.self.query.weight.detach().clone())
    init_bs.append(layer.attention.self.query.bias.detach().clone())
    init_ws.append(layer.attention.self.key.weight.detach().clone())
    init_bs.append(layer.attention.self.key.bias.detach().clone())
    init_ws.append(layer.attention.self.value.weight.detach().clone())
    init_bs.append(layer.attention.self.value.bias.detach().clone())
    init_ws.append(layer.attention.output.dense.weight.detach().clone())
    init_bs.append(layer.attention.output.dense.bias.detach().clone())
    init_ws.append(layer.attention.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.attention.output.LayerNorm.bias.detach().clone())

    init_ws.append(layer.intermediate.dense.weight.detach().clone())
    init_bs.append(layer.intermediate.dense.bias.detach().clone())
    init_ws.append(layer.output.dense.weight.detach().clone())
    init_bs.append(layer.output.dense.bias.detach().clone())
    init_ws.append(layer.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.output.LayerNorm.bias.detach().clone())

    act_cmax = act_quant_config.amax.tolist()
    wei_cmax = weight_quant_config.amax.tolist()
    init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 4)
    init_ws.append(init_clip_max)

    return init_ws, init_bs


def get_hf_bert_emb_layer_params(layer):
    init_ws = []

    init_ws.append(layer.word_embeddings.weight.detach().clone())
    init_ws.append(layer.position_embeddings.weight.detach().clone())
    init_ws.append(layer.token_type_embeddings.weight.detach().clone())
    init_ws.append(layer.LayerNorm.weight.detach().clone())
    init_ws.append(layer.LayerNorm.bias.detach().clone())

    return init_ws


def gen_bert_emb_config(training_args, config):
    bert_emb_config = BertEmbeddingLayer.get_config(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        padding_idx=config.pad_token_id,
        dropout=config.hidden_dropout_prob,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
    )
    bert_emb_config.type_vocab_size = config.type_vocab_size
    bert_emb_config.layer_norm_eps = config.layer_norm_eps
    return bert_emb_config


def inject_ls_layer(model, training_args, model_args, config):
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

    if model_args.module_type == 1 or model_args.module_type == 2:
        bert_emb_config = gen_bert_emb_config(training_args, config)
        init_ws = get_hf_bert_emb_layer_params(model.bert.embeddings)
        model.bert.embeddings = BertEmbeddingLayer(bert_emb_config, init_ws)
        if model_args.enable_quant:
            model.bert.embeddings.apply(qat_mode)
        else:
            model.bert.embeddings.apply(disable_quant)

    class LSHFTransformerEncoderLayer(TransformerEncoderLayer):
        def __init__(self, *args, **kwargs):
            super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

        def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
            ls_encoder_padding_mask = encoder_padding_mask / -10000.0
            ls_encoder_padding_mask = ls_encoder_padding_mask.squeeze()
            output = super().forward(hidden_states, ls_encoder_padding_mask)
            return (output, None, None, None)

    def gen_bert_enc_config(training_args, config):
        bert_enc_config = TransformerEncoderLayer.get_config(
            max_batch_tokens=4096,
            max_seq_len=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.num_attention_heads,
            attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
            activation_dropout_ratio=config.hidden_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            pre_layer_norm=False,
            fp16=training_args.fp16,
            local_rank=training_args.local_rank,
            activation_fn="gelu",
        )
        return bert_enc_config

    for i in range(config.num_hidden_layers):
        bert_enc_config = gen_bert_enc_config(training_args, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.bert.encoder.layer[i])
        model.bert.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_enc_config, init_ws, init_bs
        ).cuda()
        if model_args.enable_quant:
            model.bert.encoder.layer[i].apply(enable_quant)
        else:
            model.bert.encoder.layer[i].apply(disable_quant)


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

    def inject_hf_layer(config, hf_layer, ls_layer):
        for layer_id in range(config.num_hidden_layers):
            weight, bias = ls_layer[layer_id].params_dict()
            layer = hf_layer[layer_id]
            layer.attention.self.query.weight.data.copy_(weight["self_attn.q_proj"])
            layer.attention.self.query.bias.data.copy_(bias["self_attn.q_proj"])
            layer.attention.self.key.weight.data.copy_(weight["self_attn.k_proj"])
            layer.attention.self.key.bias.data.copy_(bias["self_attn.k_proj"])
            layer.attention.self.value.weight.data.copy_(weight["self_attn.v_proj"])
            layer.attention.self.value.bias.data.copy_(bias["self_attn.v_proj"])
            layer.attention.output.dense.weight.data.copy_(weight["self_attn.out_proj"])
            layer.attention.output.dense.bias.data.copy_(bias["self_attn.out_proj"])
            layer.attention.output.LayerNorm.weight.data.copy_(
                weight["self_attn_layer_norm"]
            )
            layer.attention.output.LayerNorm.bias.data.copy_(
                bias["self_attn_layer_norm"]
            )
            layer.intermediate.dense.weight.data.copy_(weight["fc1"])
            layer.intermediate.dense.bias.data.copy_(bias["fc1"])
            layer.output.dense.weight.data.copy_(weight["fc2"])
            layer.output.dense.bias.data.copy_(bias["fc2"])
            layer.output.LayerNorm.weight.data.copy_(weight["final_layer_norm"])
            layer.output.LayerNorm.bias.data.copy_(bias["final_layer_norm"])

    model_to_save = unwrap_model(model)
    if not isinstance(model_to_save, LSBertPreTrainedModel):
        raise ValueError("Must be ligtseq replaced model")
    # reload original modules
    ls_encoder_layer = model_to_save.bert.encoder.layer
    model_to_save.bert.encoder.layer = torch.nn.ModuleList(
        [BertLayer(model.config) for _ in range(model.config.num_hidden_layers)]
    )
    inject_hf_layer(
        model_to_save.config, model_to_save.bert.encoder.layer, ls_encoder_layer
    )
    state_dict = model_to_save.state_dict()
    # replace with lightseq modules
    model_to_save.bert.encoder.layer = ls_encoder_layer
    return state_dict


class LSBertPreTrainedModel(BertPreTrainedModel):
    @classmethod
    def from_pretrained(self, *args, training_args, model_args, **kwargs):
        self.config = kwargs["config"]
        model = super().from_pretrained(*args, **kwargs)
        if model_args.module_type == 1 or model_args.module_type == 2:
            inject_ls_layer(model, training_args, model_args, self.config)
        return model

    # def save_pretrained(self, *args, **kwargs):
    #     kwargs["state_dict"] = hf_state_dict(self)
    #     super().save_pretrained(*args, **kwargs)


class LSBertForSequenceClassification(
    LSBertPreTrainedModel, BertForSequenceClassification
):
    """from BertForSequenceClassification"""


class LSBertLMHeadModel(LSBertPreTrainedModel, BertLMHeadModel):
    """from BertLMHeadModel"""


class LSBertForMaskedLM(LSBertPreTrainedModel, BertForMaskedLM):
    """from BertForMaskedLM"""


class LSBertForNextSentencePrediction(
    LSBertPreTrainedModel, BertForNextSentencePrediction
):
    """from BertForNextSentencePrediction"""


class LSBertForMultipleChoice(LSBertPreTrainedModel, BertForMultipleChoice):
    """from BertForMultipleChoice"""


class LSBertForTokenClassification(LSBertPreTrainedModel, BertForTokenClassification):
    """from BertForTokenClassification"""


class LSBertForQuestionAnswering(LSBertPreTrainedModel, BertForQuestionAnswering):
    """from BertForQuestionAnswering"""
