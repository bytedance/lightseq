#include "transformer.h"

namespace lightseq {

Transformer::Transformer(const std::string weight_path,
                         const int max_batch_size)
    : LSModel({"source_ids"}, {"target_ids", "target_scores"}),
      _max_batch_size(max_batch_size) {
  /* --- step.1 initial context --- */
  Context::create_global_context(StatusType::Inference);
  _context_ptr = Context::global_instance();

  /* --- step.2 load model weights into GPU memory --- */
  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  /* --- step.4 inital operator & layer --- */

  // initial LaunchEncEmb layer
  launch_enc_emb_layer.reset(new LaunchEncEmbLayer<OpType_>(
      max_batch_tokens, tw_._padding_id, tw_._hidden_size, tw_._multilg_type));
  launch_enc_emb_layer->load_params(tw_.get_src_emb_wei(), 0);

  // // initial TransformerEncoder layers
  float attn_prob_dropout_ratio = 0.0;
  float activation_dropout_ratio = 0.0;
  float hidden_dropout_ratio = 0.0;
  int enc_wei_offset = 0;
  for (int idx = 0; idx < tw_._n_enc_layer; idx++) {
    TransformerEncoderLayerPtr<OpType_, OpType_> enc_layer_(
        new TransformerEncoderLayer<OpType_, OpType_>(
            idx, max_batch_tokens, tw_._max_step, tw_._hidden_size,
            tw_._head_num, tw_._inner_size, attn_prob_dropout_ratio,
            activation_dropout_ratio, hidden_dropout_ratio, !tw_._is_post_ln,
            tw_._use_gelu ? "gelu" : "relu", false));
    enc_wei_offset +=
        enc_layer_->load_params(tw_.get_enc_wei(), enc_wei_offset);
    enc_layer_vec.push_back(enc_layer_);
  }

  // // initial LayerNormalize layer
  enc_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      max_batch_tokens, tw_._hidden_size));
  enc_norm_layer->load_params(tw_.get_src_emb_wei(), 2);

  // initial LaunchDecEmb layer
  launch_dec_emb_layer.reset(new LaunchDecEmbLayer<OpType_>(
      max_batch_size, tw_._beam_size, tw_._hidden_size, tw_._trg_vocab_size,
      tw_._max_step, tw_._multilg_type));
  launch_dec_emb_layer->load_params(tw_.get_trg_emb_wei(), 0);

  _enc_kv_layer.reset(new EncDecKvLayer<OpType_, OpType_>(
      tw_._n_dec_layer, max_batch_tokens, tw_._hidden_size, tw_._head_num));
  _enc_kv_layer->load_params(tw_.get_trg_emb_wei(), 4);

  // initial TransformerDecoder layers
  int dec_wei_offset = 0;
  for (int idx = 0; idx < tw_._n_dec_layer; idx++) {
    TransformerDecoderLayerPtr<OpType_, OpType_> dec_layer_(
        new TransformerDecoderLayer<OpType_, OpType_>(
            tw_._n_dec_layer, idx, max_batch_tokens, tw_._max_step,
            tw_._hidden_size, tw_._head_num, tw_._inner_size, 0, 0, 0, true,
            tw_._use_gelu ? "gelu" : "relu", false, false, max_batch_size,
            tw_._beam_size));
    dec_wei_offset +=
        dec_layer_->load_params(tw_.get_dec_wei(), dec_wei_offset);
    dec_layer_vec.push_back(dec_layer_);
  }

  // // initial LayerNormalize layer
  dec_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      max_batch_size * tw_._beam_size, tw_._hidden_size));
  dec_norm_layer->load_params(tw_.get_trg_emb_wei(), 2);

  // // intial Project hidden states to vocab logits
  linear_layer.reset(new LinearLayer<OpType_, OpType_>(
      max_batch_size * tw_._beam_size, tw_._hidden_size, tw_._trg_vocab_size,
      MATRIX_OP::NonTranspose, MATRIX_OP::NonTranspose,
      tw_._no_scale_embedding ? 1.f : sqrt(1.f / tw_._hidden_size)));
  linear_layer->load_params(tw_.get_trg_emb_wei(), 0);

  sample_layer.reset(new SampleLayer<OpType_>(
      tw_._n_dec_layer, max_batch_size, tw_._max_step, tw_._trg_vocab_size,
      tw_._hidden_size, 1024, tw_._beam_size, tw_._diverse_lambda,
      tw_._dim_per_head, tw_._end_id, tw_._head_num, tw_._length_penalty));
  sample_layer->load_params(tw_.get_trg_emb_wei(), 6);

  /* --- step.5 construct network --- */
  inp_tokens = new Variable("inp_tokens", g_dtype<OpType_>());
  dec_tokens = new Variable("dec_tokens",
                            max_batch_tokens * tw_._beam_size * sizeof(int), 0,
                            VariableType::FixedVariable);
  std::tuple<Variable *, Variable *> enc_emb_outs =
      (*launch_enc_emb_layer)(inp_tokens);
  Variable *enc_emb = std::get<0>(enc_emb_outs);
  Variable *pad_mask = std::get<1>(enc_emb_outs);
  for (auto iter : enc_layer_vec) {
    enc_emb = (*iter)(enc_emb, pad_mask);
  }
  Variable *enc_out = (*enc_norm_layer)(enc_emb);

  Variable *total_enc_kv = (*_enc_kv_layer)(enc_out);

  total_enc_kv->set_regress_var();

  _context_ptr->regress_begin();
  Variable *dec_emb = (*launch_dec_emb_layer)(dec_tokens);
  cache_size =
      max_batch_tokens * tw_._beam_size * tw_._hidden_size * sizeof(OpType_);
  total_cache_k = new Variable("total_cache_k", cache_size * tw_._n_dec_layer,
                               0, VariableType::RegressiveVariable);
  total_cache_v = new Variable("total_cache_v", cache_size * tw_._n_dec_layer,
                               0, VariableType::RegressiveVariable);
  pad_mask->set_regress_var();

  int dec_layer_idx = 0;
  for (auto iter : dec_layer_vec) {
    Variable *cache_k =
        new Variable("cache_k", total_cache_k, cache_size * dec_layer_idx);
    Variable *cache_v =
        new Variable("cache_v", total_cache_v, cache_size * dec_layer_idx);
    std::tuple<Variable *, Variable *, Variable *> dec_outs =
        (*iter)(dec_emb, total_enc_kv, pad_mask, cache_k, cache_v);
    dec_emb = std::get<0>(dec_outs);
    Variable *cache_k_out = std::get<1>(dec_outs);
    Variable *cache_v_out = std::get<2>(dec_outs);

    dec_layer_idx++;
  }
  Variable *dec_out = (*dec_norm_layer)(dec_emb);
  dec_out = (*linear_layer)(dec_out);

  std::tuple<Variable *, Variable *> sample_outs =
      (*sample_layer)(dec_out, dec_tokens, total_cache_k, total_cache_v);
  _context_ptr->regress_end();

  dec_tokens_buf = std::get<0>(sample_outs);
  seq_score = std::get<1>(sample_outs);
  dec_tokens_buf->malloc_memory(max_batch_tokens * tw_._beam_size *
                                sizeof(int));

  transformer_out = new Variable("transformer_out");

  std::vector<int> start_id_vec(
      _max_batch_size * tw_._beam_size * tw_._max_step, tw_._start_id);
  CHECK_GPU_ERROR(cudaMemcpyAsync(dec_tokens->value(), start_id_vec.data(),
                                  sizeof(int) * start_id_vec.size(),
                                  cudaMemcpyHostToDevice,
                                  _context_ptr->get_stream()));
}

Transformer::~Transformer() {}

void Transformer::encoder_before_forward(int batch_size, int seq_len) {
  launch_enc_emb_layer->before_forward(batch_size, seq_len);
  int dec_layer_idx = 0;
  for (auto iter : enc_layer_vec) {
    iter->before_forward(batch_size, seq_len);
    dec_layer_idx++;
  }
  enc_norm_layer->before_forward(batch_size * seq_len);
  _enc_kv_layer->before_forward(batch_size, seq_len);
}

void Transformer::decoder_before_forward(int batch_size, int seq_len,
                                         int cur_step) {
  launch_dec_emb_layer->before_forward(batch_size, cur_step);

  for (auto iter : dec_layer_vec) {
    iter->before_forward(batch_size, tw_._beam_size, seq_len, cur_step);
  }

  int beam_batch_size = batch_size * tw_._beam_size;
  dec_norm_layer->before_forward(beam_batch_size);
  linear_layer->before_forward(beam_batch_size, 1);
  sample_layer->before_forward(batch_size, cur_step);
}

void Transformer::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  if (tw_._sampling_method == "topk" || tw_._sampling_method == "topp") {
    _output_topk = false;
  }
  if (tw_._sampling_method == "topk_greedy") {
    _output_topk = true;
  }

  int _batch_max_decode_length =
      std::min(tw_._max_step, seq_len + tw_._extra_decode_length) - 1;

  _is_sampling =
      (tw_._sampling_method == "topk" || tw_._sampling_method == "topp" ||
       tw_._sampling_method == "topk_greedy");

  if (_is_sampling) {
    _batch_max_decode_length = tw_._max_step;
  }

  /* --- notice that the order of forward should be the same with network --- */
  encoder_before_forward(batch_size, seq_len);
  decoder_before_forward(batch_size, seq_len, 0);

  launch_enc_emb_layer->forward();
  for (auto iter : enc_layer_vec) {
    iter->forward();
  }
  enc_norm_layer->forward();
  _enc_kv_layer->forward();

  int step = 0;
  for (step = 0; step < _batch_max_decode_length - 1; step++) {
    decoder_before_forward(batch_size, seq_len, step);

    launch_dec_emb_layer->forward();
    for (auto iter : dec_layer_vec) {
      iter->forward();
    }
    dec_norm_layer->forward();
    linear_layer->forward();
    sample_layer->forward();
    if (sample_layer->is_stop()) {
      break;
    }
    Variable::swap_tensor(dec_tokens, dec_tokens_buf);
  }

  if (_output_topk || _is_sampling) {
    ker_write_topk_result<<<batch_size * tw_._beam_size, step + 1, 0,
                            _context_ptr->get_stream()>>>(
        (int *)dec_tokens->value(), (float *)seq_score->value(),
        (int *)transformer_out->value(), tw_._trg_vocab_size, tw_._max_step,
        tw_._beam_size, tw_._end_id);
  } else {
    if (tw_._length_penalty >= 0.f || step == _batch_max_decode_length) {
      ker_write_trg_tokenid_pos_penalty<<<batch_size, step + 1, 0,
                                          _context_ptr->get_stream()>>>(
          (int *)dec_tokens->value(), (float *)seq_score->value(),
          (int *)transformer_out->value(), tw_._max_step, tw_._beam_size);
    } else {
      ker_write_trg_tokenid_neg_penalty<<<batch_size, step + 1, 0,
                                          _context_ptr->get_stream()>>>(
          (int *)dec_tokens->value(), (float *)seq_score->value(),
          (int *)transformer_out->value(), tw_._max_step, tw_._beam_size,
          tw_._trg_vocab_size, tw_._end_id);
    }
  }
  /* ---step3. output the decoding result--- */

  _context_ptr->synchronize();

  set_output_shape(0,
                   {batch_size, _output_topk ? tw_._beam_size : 1, step + 1});
  set_output_shape(1, {batch_size, _output_topk ? tw_._beam_size : 1});
}

void Transformer::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      inp_tokens->set_value(static_cast<char *>(input_ptr));
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Transformer::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      transformer_out->set_value(static_cast<char *>(output_ptr));
      break;

    case 1:
      seq_score->set_value(static_cast<char *>(output_ptr));
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
const void *Transformer::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(transformer_out->value());

    case 1:
      return static_cast<void *>(seq_score->value());

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Transformer::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

std::vector<int> Transformer::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._beam_size, tw_._max_step};
      break;

    case 1:
      return {_max_batch_size, tw_._beam_size};
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Transformer::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Transformer::get_output_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    case 1:
      return DataType::kFloat32;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace lightseq
