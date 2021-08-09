from fairseq.incremental_decoding_utils import with_incremental_state
from lightseq.training.ops.pytorch.transformer_decoder_layer import (
    LSTransformerDecoderLayer,
)


@with_incremental_state
class LSFSTransformerDecoderLayer(LSTransformerDecoderLayer):
    """Decoder layer only for inference."""

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super().__init__(config, initial_weights, initial_biases)

    def get_self_attn_cache(self, incremental_state):
        res = self.get_incremental_state(incremental_state, "cache")
        if res is not None:
            return res
        else:
            return {}

    def set_self_attn_cache(self, incremental_state, cache):
        self.set_incremental_state(incremental_state, "cache", cache)

    def reorder_incremental_state(self, incremental_state, new_order):
        cache = self.get_self_attn_cache(incremental_state)
        if cache is not None:
            for k in cache.keys():
                if k == "encdec_kv":
                    cur_order = new_order // self.beam_size
                    cur_order = cur_order[:: self.beam_size]
                    idx = 1
                else:
                    cur_order = new_order
                    idx = 0
                value = cache[k]
                cache[k] = value.index_select(idx, cur_order)
            self.set_self_attn_cache(incremental_state, cache)

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
    ):
        if incremental_state is None:
            cache = None
        else:
            cache = self.get_self_attn_cache(incremental_state)
        self.beam_size = int(x.shape[0] / encoder_padding_mask.shape[0])
        res = super().forward(x, encoder_out, encoder_padding_mask, cache)
        if cache:
            self.set_self_attn_cache(incremental_state, cache)
        return res, None, None
