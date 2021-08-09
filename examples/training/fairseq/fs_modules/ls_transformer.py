import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import LayerNorm
from lightseq.training.ops.pytorch.transformer_embedding_layer import (
    LSTransformerEmbeddingLayer,
)
from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer,
)

from .ls_fs_transformer_decoder_layer import LSFSTransformerDecoderLayer

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)
MAX_SEQ_LENGTH = 300


@register_model("ls_transformer")
class LSTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )
        # fmt: on

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
            max_seq_len=max_positions,
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
            max_seq_len=MAX_SEQ_LENGTH,
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
        self.batch_size = x.shape[0]
        self.beam_size = -1

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
        if self.beam_size < 0:
            self.beam_size = int(new_order.shape[0] / self.batch_size)
        else:
            new_order = new_order // self.beam_size
        new_order = new_order[:: self.beam_size]
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
        self.embed_tokens = embed_tokens
        self.padding_idx = self.embed_tokens.config.padding_idx

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
            max_seq_len=MAX_SEQ_LENGTH,
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


@register_model_architecture("ls_transformer", "ls_transformer_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    return base_architecture(args)


@register_model_architecture("ls_transformer", "ls_transformer")
def base_architecture(args):
    # specify a small value (300) which meet the needs of most NLP datasets, to avoid OOM error
    args.max_source_positions = min(
        MAX_SEQ_LENGTH, getattr(args, "max_source_positions", MAX_SEQ_LENGTH)
    )
    args.max_target_positions = min(
        MAX_SEQ_LENGTH, getattr(args, "max_target_positions", MAX_SEQ_LENGTH)
    )

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("ls_transformer", "ls_transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("ls_transformer", "ls_transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("ls_transformer", "ls_transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)
