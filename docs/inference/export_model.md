## Export model
In order to serve your own model, you need to export model trained from deeplearning framework(E.g.
TenforFlow, PyTorch) to custom model proto defined by LightSeq. The following explains the custom
transformer model format.

### transformer model proto
``` c
syntax = "proto3";
option optimize_for = LITE_RUNTIME;
// all the matrix are stored in row-major order,
// plz see https://en.wikipedia.org/wiki/Row-_and_column-major_order for details

// the definition of "Multi-Head Attention", "Scaled Dot-Product Attention" and
// "Feed-Forward Networks"
// plz see https://arxiv.org/abs/1706.03762 for details

message EncoderLayer {
  // encoder-self-attention
  repeated float multihead_norm_scale = 1;  // [hidden_size]
  repeated float multihead_norm_bias = 2;   // [hidden_size]
  // "Multi-Head Attention" linearly project weights kernel for query, key,
  // value,
  // before "Scaled Dot-Product Attention, with shape (hidden_size,
  // hidden_size*3)
  // is built by numpy.concatenate((query_kernel, key_kernel, value_kernel),
  // axis=1)
  // perform numpy.dot(input, multihead_project_kernel_qkv) will get the [query,
  // key, value] of
  // "Scaled Dot-Product Attention"
  repeated float multihead_project_kernel_qkv =
      3;  // [hidden_size, 3, hidden_size]
  repeated float multihead_project_bias_qkv = 4;  // [3, hidden_size]
  // "Multi-Head Attention" linearly project weights kernel for output
  // after "Scaled Dot-Product Attention", with shape (hidden_size, hidden_size)
  repeated float multihead_project_kernel_output =
      5;  // [hidden_size, hidden_size]
  repeated float multihead_project_bias_output = 6;  // [hidden_size]

  // "Feed-Forward Networks"
  repeated float ffn_norm_scale = 7;      // [hidden_size]
  repeated float ffn_norm_bias = 8;       // [hidden_size]
  repeated float ffn_first_kernel = 9;    // [hidden_size, inner_size]
  repeated float ffn_first_bias = 10;     // [inner_size]
  repeated float ffn_second_kernel = 11;  // [inner_size, hidden_size]
  repeated float ffn_second_bias = 12;    // [hidden_size]
}

message DecoderLayer {
  // decoder-self-attention
  repeated float self_norm_scale = 1;          // [hidden_size]
  repeated float self_norm_bias = 2;           // [hidden_size]
  repeated float self_project_kernel_qkv = 3;  // [hidden_size, 3, hidden_size]
  repeated float self_project_bias_qkv = 4;    // [3, hidden_size]
  repeated float self_project_kernel_output = 5;  // [hidden_size, hidden_size]
  repeated float self_project_bias_output = 6;    // [hidden_size]

  // decoder-encode-attention
  repeated float encdec_norm_scale = 7;        // [hidden_size]
  repeated float encdec_norm_bias = 8;         // [hidden_size]
  repeated float encdec_project_kernel_q = 9;  // [hidden_size, hidden_size]
  repeated float encdec_project_bias_q = 10;   // [hidden_size]
  repeated float encdec_project_kernel_output =
      11;                                          // [hidden_size, hidden_size]
  repeated float encdec_project_bias_output = 12;  // [hidden_size]

  // "Feed-Forward Networks"
  repeated float ffn_norm_scale = 13;     // [hidden_size]
  repeated float ffn_norm_bias = 14;      // [hidden_size]
  repeated float ffn_first_kernel = 15;   // [hidden_size, inner_size]
  repeated float ffn_first_bias = 16;     // [inner_size]
  repeated float ffn_second_kernel = 17;  // [inner_size, hidden_size]
  repeated float ffn_second_bias = 18;    // [hidden_size]
}

message EmbeddingLayer {
  // token embedding table
  // for encoder, it is in [src_vocab_size, hidden_size]
  // for decoder, it is in [hidden_size, trg_vocab_size]
  // notice, it shoule have been multiply by sqrt(hidden_size)
  // so, look it up directly will get the input token embedding, there is no
  // need
  // to multiply by sqrt(hidden_size) during inference.
  repeated float token_embedding = 1;
  repeated float position_embedding = 2;  // [max_step, hidden_size]
  // the last layer_norm of encoder or decoder
  repeated float norm_scale = 3;  // [hidden_size]
  repeated float norm_bias = 4;   // [hidden_size]

  // below only for trg, not in src
  // [hidden_size, enc_layer, 2, hidden_size]
  repeated float encode_output_project_kernel_kv = 5;
  // [enc_layer, 2, hidden_size]
  repeated float encode_output_project_bias_kv = 6;
  // decoder vocab logit bias
  repeated float shared_bias = 7;  // [target_vocab_size]
}

message ModelConf {
  int32 head_num = 1;   // head number for multi-head attention
  int32 beam_size = 2;  // beam size of beam search
  int32 extra_decode_length =
      3;                     // extra decode length compared with source length
  float length_penalty = 4;  // length penalty of beam search
  int32 src_padding_id = 5;  // source padding id
  int32 trg_start_id = 6;    // target start id
}

message Transformer {
  EmbeddingLayer src_embedding = 1;         // source embedding
  repeated EncoderLayer encoder_stack = 2;  // encoder weights
  EmbeddingLayer trg_embedding = 3;         // target embedding
  repeated DecoderLayer decoder_stack = 4;  // decoder weighs
  ModelConf model_conf = 5;                 // model_config
}
```
So you only need to extract the model weights from deeplearning framework and fill it into proto, then you can serve your
own model.
