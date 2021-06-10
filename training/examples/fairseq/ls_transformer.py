import torch
import torch.nn as nn
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import LayerNorm
from ops.pytorch.transformer_encoder_layer import LSTransformerEncoderLayer
from ops.pytorch.transformer_embedding_layer import LSTransformerEmbeddingLayer
from examples.fairseq.ls_fs_transformer_decoder_layer import LSFSTransformerDecoderLayer


class LSTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, task):
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens = cls.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.max_source_positions
        )
        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_embed_dim, args.max_target_positions
        )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, max_positions, **kwargs):
        config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=len(dictionary),
            embedding_dim=embed_dim,
            max_batch_tokens=args.max_tokens,
            max_seq_len=256,  # FIXME later
            padding_idx=dictionary.pad(),
            dropout=args.dropout,
            fp16=args.fp16,
            local_rank=args.device_id,
        )
        emb = LSTransformerEmbeddingLayer(config)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LSTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LSTransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class LSTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)

        embed_dim = args.encoder_embed_dim
        self.embed_tokens = embed_tokens
        self.padding_idx = self.embed_tokens.config.padding_idx

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = LayerNorm(embed_dim)

    def build_encoder_layer(self, args):
        config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=args.max_tokens,
            max_seq_len=256,
            hidden_size=args.encoder_embed_dim,
            intermediate_size=args.encoder_ffn_embed_dim,
            nhead=args.encoder_attention_heads,
            attn_prob_dropout_ratio=args.attention_dropout,
            activation_dropout_ratio=args.activation_dropout,
            hidden_dropout_ratio=args.dropout,
            pre_layer_norm=args.encoder_normalize_before,
            fp16=args.fp16,
            local_rank=args.device_id,
        )
        return LSTransformerEncoderLayer(config)

    def forward_embedding(self, src_tokens):
        x = self.embed_tokens(src_tokens)
        return x

    def forward(self, src_tokens, **kwargs):
        x = self.forward_embedding(src_tokens)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        # x: [batch_size, seq_len, hidden_size]
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = self.layer_norm(x)
        # self.batch_size = x.shape[0]
        # self.beam_size = -1

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.args.max_source_positions

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        if self.beam_size < 0:
            self.beam_size = int(new_order.shape[0] / self.batch_size)
        else:
            new_order = new_order // self.beam_size
        new_order = new_order[:: self.beam_size]
        new_encoder_out = encoder_out.encoder_out.index_select(1, new_order)
        new_encoder_padding_mask = encoder_out.encoder_padding_mask.index_select(
            0, new_order
        )
        """
        new_encoder_out = encoder_out.encoder_out.index_select(1, new_order)
        new_encoder_padding_mask = encoder_out.encoder_padding_mask.index_select(
            0, new_order
        )

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class LSTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, **kwargs):
        self.args = args
        super().__init__(dictionary)

        self._future_mask = torch.empty(0)

        embed_dim = args.decoder_embed_dim
        self.padding_idx = embed_tokens.config.padding_idx
        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList(
            [self.build_decoder_layer(args) for _ in range(args.decoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = LayerNorm(embed_dim)

        self.output_projection = nn.Linear(
            self.embed_tokens.embeddings.shape[1],
            self.embed_tokens.embeddings.shape[0],
            bias=False,
        )
        self.output_projection.weight = self.embed_tokens.embeddings

    def build_decoder_layer(self, args):
        config = LSFSTransformerDecoderLayer.get_config(
            max_batch_tokens=args.max_tokens,
            max_seq_len=256,
            hidden_size=args.decoder_embed_dim,
            intermediate_size=args.decoder_ffn_embed_dim,
            nhead=args.decoder_attention_heads,
            attn_prob_dropout_ratio=args.attention_dropout,
            activation_dropout_ratio=args.activation_dropout,
            hidden_dropout_ratio=args.dropout,
            pre_layer_norm=args.decoder_normalize_before,
            fp16=args.fp16,
            local_rank=args.device_id,
            nlayer=args.decoder_layers,
        )
        return LSFSTransformerDecoderLayer(config)

    def forward_embedding(self, prev_output_tokens, incremental_state=None):
        step = 0
        if incremental_state is not None:
            step = prev_output_tokens.size(1) - 1
            prev_output_tokens = prev_output_tokens[:, -1:]

        x = self.embed_tokens(prev_output_tokens, step)
        return x, prev_output_tokens

    def forward(
        self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs
    ):
        x, prev_output_tokens = self.forward_embedding(
            prev_output_tokens, incremental_state
        )

        # x: [batch_size, seq_len, hidden_size]
        for _, layer in enumerate(self.layers):
            x, _, _ = layer(
                x,
                encoder_out.encoder_out,
                encoder_out.encoder_padding_mask,
                incremental_state,
            )

        x = self.layer_norm(x)

        x = self.output_projection(x)
        return x, None

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.args.max_target_positions
