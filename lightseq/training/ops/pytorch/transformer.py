from dataclasses import dataclass

import torch
import torch.nn as nn

from lightseq.training import (
    LSTransformerEmbeddingLayer,
    LSTransformerEncoderLayer,
    LSTransformerDecoderLayer,
)
from lightseq.training.ops.pytorch.util import MODEL_ARCH


class LSTransformer(nn.Module):
    def __init__(self, config):
        super(LSTransformer, self).__init__()
        self.config = config

        print("Lightseq Transformer config is ", self.config.__dict__)

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.build_model(self.config)

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            vocab_size: int  # vocabulary size
            padding_idx: int  # index of padding token
            num_encoder_layer: int  # number of encoder layer
            num_decoder_layer: int  # number of decoder layer
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            activation_fn: str  # relu or gelu
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        return Config(**kwargs)

    def build_model(self, config):
        encoder_embed_tokens = self.build_embedding(config)
        decoder_embed_tokens = self.build_embedding(config)

        self.encoder = self.build_encoder(config, encoder_embed_tokens)
        self.decoder = self.build_decoder(config, decoder_embed_tokens)

    def build_embedding(self, config):
        emb_config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            padding_idx=config.padding_idx,
            dropout=config.hidden_dropout_ratio,
            fp16=config.fp16,
            local_rank=config.local_rank,
        )
        emb = LSTransformerEmbeddingLayer(emb_config)
        return emb

    def build_encoder(self, config, embed_tokens):
        return LSTransformerEncoder(config, embed_tokens)

    def build_decoder(self, config, embed_tokens):
        return LSTransformerDecoder(config, embed_tokens)

    def forward(self, src_tokens, trg_tokens):
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        decoder_out = self.decoder(trg_tokens, encoder_out, encoder_padding_mask)
        return decoder_out


class LSTransformerEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super(LSTransformerEncoder, self).__init__()
        self.config = config

        embed_dim = embed_tokens.config.embedding_dim
        self.embed_tokens = embed_tokens
        self.padding_idx = self.config.padding_idx

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(config) for _ in range(config.num_encoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def build_encoder_layer(self, config):
        enc_config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
        )
        return LSTransformerEncoderLayer(enc_config)

    def forward_embedding(self, src_tokens):
        x = self.embed_tokens(src_tokens)
        return x

    def forward(self, src_tokens):
        x = self.forward_embedding(src_tokens)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return x, encoder_padding_mask


class LSTransformerDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super(LSTransformerDecoder, self).__init__()
        self.config = config

        embed_dim = embed_tokens.config.embedding_dim
        self.embed_tokens = embed_tokens
        self.padding_idx = self.config.padding_idx

        self.layers = nn.ModuleList(
            [self.build_decoder_layer(config) for _ in range(config.num_decoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Linear(
            self.embed_tokens.embeddings.shape[1],
            self.embed_tokens.embeddings.shape[0],
            bias=False,
        )
        self.output_projection.weight = self.embed_tokens.embeddings

    def build_decoder_layer(self, config):
        dec_config = LSTransformerDecoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
            nlayer=config.num_decoder_layer,
        )
        return LSTransformerDecoderLayer(dec_config)

    def forward_embedding(self, trg_tokens, cache=None):
        step = 0
        if cache is not None:
            step = trg_tokens.size(1) - 1
            trg_tokens = trg_tokens[:, -1:]
        x = self.embed_tokens(trg_tokens, step)
        return x

    def forward(self, trg_tokens, encoder_out, encoder_padding_mask, cache=None):
        x = self.forward_embedding(trg_tokens, cache)

        if cache == {}:
            for i in range(self.num_layers):
                cache[i] = {}

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x = layer(
                x,
                encoder_out,
                encoder_padding_mask,
                layer_cache,
            )

        x = self.layer_norm(x)

        x = self.output_projection(x)
        return x
