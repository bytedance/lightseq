from dataclasses import dataclass
from torch import nn
from lightseq.training.ops.pytorch.util import MODEL_ARCH, check_config


class TransformerEncoderLayerBase(nn.Module):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            activation_fn: str = "relu"  # relu or gelu

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        config = Config(**kwargs)
        check_config(config)
        return config


class TransformerDecoderLayerBase(nn.Module):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            nlayer: int  # number of layers
            activation_fn: str = "relu"  # relu or gelu
            has_cross_attn: bool = True

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        config = Config(**kwargs)
        check_config(config)
        return config


class TransformerEmbeddingLayerBase(nn.Module):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            vocab_size: int  # vocabulary size
            embedding_dim: int  # embedding size
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            padding_idx: int  # padding token id in vocabulary
            dropout: float  # embedding dropout ration
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            trainable_pos: bool = False  # trainable positional embedding
            no_scale_embedding: bool = False  # scale embedding
            layernorm_embedding: bool = False  # layernorm for embedding
            need_offset: bool = False  # position offset for bart

        return Config(**kwargs)
