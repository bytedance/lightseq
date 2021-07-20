import torch

from tests.fairseq_layers import (
    FSTransformerEncoderLayer,
    FSTransformerDecoderLayer,
    FSTransformerEmbeddingLayer,
    FSCrossEntropyLayer,
    get_fairseq_enc_params,
    get_fairseq_dec_params,
)
from lightseq.training import (
    LSTransformerEncoderLayer,
    LSTransformerEmbeddingLayer,
    LSCrossEntropyLayer,
)
from examples.training.fairseq.fs_modules.ls_fs_transformer_decoder_layer import (
    LSFSTransformerDecoderLayer,
)


###################### encoder layer ######################
def gen_enc_layer(global_config):
    def gen_ls_enc_layer(initial_weights=None, initial_biases=None):
        config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=global_config.max_batch_tokens,
            max_seq_len=global_config.max_seq_len,
            hidden_size=global_config.hidden_size,
            intermediate_size=global_config.intermediate_size,
            nhead=global_config.nhead,
            attn_prob_dropout_ratio=global_config.attn_prob_dropout_ratio,
            activation_dropout_ratio=global_config.activation_dropout_ratio,
            hidden_dropout_ratio=global_config.hidden_dropout_ratio,
            pre_layer_norm=global_config.pre_layer_norm,
            fp16=global_config.fp16,
            local_rank=global_config.local_rank,
            activation_fn=global_config.activation_fn,
        )
        layer = LSTransformerEncoderLayer(config, initial_weights, initial_biases)
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    def gen_fs_enc_layer():
        layer = FSTransformerEncoderLayer(
            embed_dim=global_config.hidden_size,
            ffn_embed_dim=global_config.intermediate_size,
            nhead=global_config.nhead,
            dropout=global_config.hidden_dropout_ratio,
            attn_dropout=global_config.attn_prob_dropout_ratio,
            activation_dropout=global_config.activation_dropout_ratio,
            normalize_before=global_config.pre_layer_norm,
            activation_fn=global_config.activation_fn,
        )
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    custom_enc_layer_list = []
    fairseq_enc_layer_list = []

    for _ in range(global_config.num_layers):
        fairseq_enc_layer = gen_fs_enc_layer()
        initial_enc_weights, initial_enc_biases = get_fairseq_enc_params(
            fairseq_enc_layer
        )
        custom_enc_layer = gen_ls_enc_layer(initial_enc_weights, initial_enc_biases)
        custom_enc_layer_list.append(custom_enc_layer)
        fairseq_enc_layer_list.append(fairseq_enc_layer)

    return torch.nn.ModuleList(custom_enc_layer_list), torch.nn.ModuleList(
        fairseq_enc_layer_list
    )


###################### decoder layer ######################
def gen_dec_layer(global_config):
    def gen_ls_dec_layer(initial_weights=None, initial_biases=None):
        config = LSFSTransformerDecoderLayer.get_config(
            max_batch_tokens=global_config.max_batch_tokens,
            max_seq_len=global_config.max_seq_len,
            hidden_size=global_config.hidden_size,
            intermediate_size=global_config.intermediate_size,
            nhead=global_config.nhead,
            attn_prob_dropout_ratio=global_config.attn_prob_dropout_ratio,
            activation_dropout_ratio=global_config.activation_dropout_ratio,
            hidden_dropout_ratio=global_config.hidden_dropout_ratio,
            pre_layer_norm=global_config.pre_layer_norm,
            fp16=global_config.fp16,
            local_rank=global_config.local_rank,
            nlayer=global_config.num_layers,
            activation_fn=global_config.activation_fn,
        )
        layer = LSFSTransformerDecoderLayer(
            config,
            initial_weights,
            initial_biases,
        )
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    def gen_fs_dec_layer():
        layer = FSTransformerDecoderLayer(
            embed_dim=global_config.hidden_size,
            ffn_embed_dim=global_config.intermediate_size,
            nhead=global_config.nhead,
            encoder_embed_dim=global_config.hidden_size,
            dropout=global_config.hidden_dropout_ratio,
            attn_dropout=global_config.attn_prob_dropout_ratio,
            activation_dropout=global_config.activation_dropout_ratio,
            normalize_before=global_config.pre_layer_norm,
            activation_fn=global_config.activation_fn,
        )
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    custom_dec_layer_list = []
    fairseq_dec_layer_list = []
    _initial_dec_weights_list = []
    _initial_dec_biases_list = []
    _initial_encdec_attn_kvw_list = []
    _initial_encdec_attn_kvb_list = []

    for _ in range(global_config.num_layers):
        fairseq_dec_layer = gen_fs_dec_layer()
        initial_dec_weights, initial_dec_biases = get_fairseq_dec_params(
            fairseq_dec_layer
        )
        fairseq_dec_layer_list.append(fairseq_dec_layer)
        _initial_dec_weights_list.append(initial_dec_weights)
        _initial_dec_biases_list.append(initial_dec_biases)
        _initial_encdec_attn_kvw_list.append(initial_dec_weights[6])
        _initial_encdec_attn_kvw_list.append(initial_dec_weights[7])
        _initial_encdec_attn_kvb_list.append(initial_dec_biases[6])
        _initial_encdec_attn_kvb_list.append(initial_dec_biases[7])

    _initial_encdec_attn_kvw = torch.cat(_initial_encdec_attn_kvw_list, dim=0)
    _initial_encdec_attn_kvb = torch.cat(_initial_encdec_attn_kvb_list, dim=0)
    for i in range(global_config.num_layers):
        _initial_dec_weights_list[i].pop(7)
        _initial_dec_weights_list[i].pop(6)
        if i == 0:
            _initial_dec_weights_list[i].append(_initial_encdec_attn_kvw)
        _initial_dec_biases_list[i].pop(7)
        _initial_dec_biases_list[i].pop(6)
        if i == 0:
            _initial_dec_biases_list[i].append(_initial_encdec_attn_kvb)
        custom_dec_layer = gen_ls_dec_layer(
            _initial_dec_weights_list[i], _initial_dec_biases_list[i]
        )
        custom_dec_layer_list.append(custom_dec_layer)

    return torch.nn.ModuleList(custom_dec_layer_list), torch.nn.ModuleList(
        fairseq_dec_layer_list
    )


###################### embedding layer ######################
def gen_emb_layer(global_config):
    def gen_ls_emb_layer(initial_embedding=None):
        config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=global_config.vocab_size,
            embedding_dim=global_config.hidden_size,
            max_batch_tokens=global_config.max_batch_tokens,
            max_seq_len=global_config.max_seq_len,
            padding_idx=global_config.padding_idx,
            dropout=global_config.hidden_dropout_ratio,
            fp16=global_config.fp16,
            local_rank=global_config.local_rank,
        )
        layer = LSTransformerEmbeddingLayer(config, initial_embedding)
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    def gen_fs_emb_layer():
        layer = FSTransformerEmbeddingLayer(
            vocab_size=global_config.vocab_size,
            embedding_dim=global_config.hidden_size,
            max_seq_len=global_config.max_seq_len,
            padding_idx=global_config.padding_idx,
            dropout=global_config.hidden_dropout_ratio,
            fp16=global_config.fp16,
        )
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    fairseq_emb_layer = gen_fs_emb_layer()
    initial_embedding = fairseq_emb_layer.embeddings.weight.detach().clone()
    custom_emb_layer = gen_ls_emb_layer(initial_embedding)

    return custom_emb_layer, fairseq_emb_layer


###################### cross entropy layer ######################
def gen_ce_layer(global_config):
    def gen_ls_ce_layer():
        config = LSCrossEntropyLayer.get_config(
            max_batch_tokens=global_config.max_batch_tokens,
            padding_idx=global_config.padding_idx,
            epsilon=global_config.label_smooth,
            fp16=global_config.fp16,
            local_rank=global_config.local_rank,
        )
        layer = LSCrossEntropyLayer(config)
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    def gen_fs_ce_layer():
        layer = FSCrossEntropyLayer(
            epsilon=global_config.label_smooth,
            ignore_index=global_config.padding_idx,
        )
        layer.to(torch.device("cuda:{}".format(global_config.local_rank)))
        layer.train()
        return layer

    fairseq_ce_layer = gen_fs_ce_layer()
    custom_ce_layer = gen_ls_ce_layer()

    return custom_ce_layer, fairseq_ce_layer
