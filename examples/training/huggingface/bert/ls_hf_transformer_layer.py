from lightseq.training.pytorch_quantization.nn.modules.tensor_quantizer import (
    enable_quant,
)
from lightseq.training.ops.pytorch.quantization import qat_mode, disable_quant
from lightseq.training.ops.pytorch.torch_transformer_layers import BertEmbeddingLayer


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

    if model_args.module_type == 2:
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
