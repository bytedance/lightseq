#include "gpt.h"

namespace lightseq {
namespace cuda {
Gpt::Gpt(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"gpt_out", "gpt_scores"}),
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
  printf("*** model max_batch_size: %d ***\n", max_batch_size);
  _generate_method = get_generate_method(tw_._sampling_method);
  if (_generate_method != GenerateMethod::BeamSearch) {
    tw_._beam_size = 1;
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  _inp_tokens = new Variable("inp_tokens", g_dtype<OpType_>());

  /* --- step.4 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  // initial LaunchEncEmb layer
  _launch_gpt_emb_layer.reset(new LaunchGptEmbLayer<OpType_>(
      max_batch_tokens, tw_._max_step, _max_batch_size, tw_._beam_size, tw_._padding_id,
      tw_._hidden_size));
  _launch_gpt_emb_layer->load_params(tw_.get_src_emb_wei(), 0);

  // initial TransformerEncoder layers
  float attn_prob_dropout_ratio = 0.0;
  float activation_dropout_ratio = 0.0;
  float hidden_dropout_ratio = 0.0;
  int enc_wei_offset = 0;
  for (int idx = 0; idx < tw_._n_enc_layer; idx++) {
    GptLayerPtr<OpType_, OpType_> gpt_layer(new GptLayer<OpType_, OpType_>(
        idx, max_batch_tokens * tw_._beam_size, tw_._max_step, tw_._hidden_size,
        tw_._head_num, tw_._inner_size, attn_prob_dropout_ratio,
        activation_dropout_ratio, hidden_dropout_ratio,
        tw_._use_gelu ? "gelu" : "relu", false));
    enc_wei_offset += gpt_layer->load_params(tw_.get_enc_wei(), enc_wei_offset);
    _gpt_layers_vec.push_back(gpt_layer);
  }

  // initial LayerNormalize layer
  _lyr_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      max_batch_size * tw_._beam_size, tw_._hidden_size));
  _lyr_norm_layer->load_params(tw_.get_src_emb_wei(), 2);

  // intial Project hidden states to vocab logits
  _linear_layer.reset(new LinearLayer<OpType_, OpType_>(
      max_batch_size * tw_._beam_size, tw_._hidden_size, tw_._src_vocab_size,
      MATRIX_OP::Transpose, MATRIX_OP::NonTranspose, 1.f));
  _linear_layer->load_params(tw_.get_src_emb_wei(), 0);

  _generator_layer.reset(new GeneratorLayer<OpType_>(
      _generate_method, tw_._n_enc_layer, max_batch_size, tw_._max_step,
      tw_._src_vocab_size, tw_._hidden_size, 1024, tw_._beam_size,
      tw_._diverse_lambda, tw_._dim_per_head, tw_._eos_id, tw_._head_num,
      tw_._length_penalty, tw_._topk, tw_._topp, false));

  printf("Finish initialize layers and assign weights!\n");

  /* --- step.5 construct network --- */
  size_t cache_size = max_batch_tokens * tw_._beam_size * tw_._hidden_size;
  _total_caches_k = new Variable(
      "total_caches_k", cache_size * tw_._n_enc_layer, g_dtype<OpType_>(),
      DataType::kNotSupported, VariableType::RegressiveVariable);
  _total_caches_v = new Variable(
      "total_caches_v", cache_size * tw_._n_enc_layer, g_dtype<OpType_>(),
      DataType::kNotSupported, VariableType::RegressiveVariable);

  // note regress begin
  _context_ptr->regress_begin();

  std::tuple<Variable *, Variable *, Variable*> gpt_emb_outs =
      (*_launch_gpt_emb_layer)(_inp_tokens);
  Variable *gpt_emb = std::get<0>(gpt_emb_outs);
  _pad_mask = std::get<1>(gpt_emb_outs);
  _pad_mask->set_regress_var();
  size_t cache_offset = 0;
  for (auto iter : _gpt_layers_vec) {
    Variable *cache_k = new Variable("cache_k", _total_caches_k);
    cache_k->set_offset(cache_offset, {cache_size});
    Variable *cache_v = new Variable("cache_v", _total_caches_v);
    cache_v->set_offset(cache_offset, {cache_size});
    gpt_emb = (*iter)(gpt_emb, cache_k, cache_v, _pad_mask);
    cache_offset += cache_size;
  }
  gpt_emb = (*_lyr_norm_layer)(gpt_emb);
  Variable *logits_prob = (*_linear_layer)(gpt_emb);

  std::tuple<Variable *, Variable *> gen_outs =
      (*_generator_layer)(logits_prob, _inp_tokens);

  // note regress_end
  _context_ptr->regress_end();

  _out_tokens = std::get<0>(gen_outs);
  _out_scores = std::get<1>(gen_outs);
  _inp_tokens->malloc_memory(_max_batch_size * tw_._beam_size * tw_._max_step);
  _out_tokens->malloc_memory(_max_batch_size * tw_._beam_size * tw_._max_step);

  _context_ptr->build();
  printf("Finish construct network!\n");
}

Gpt::~Gpt() {}

void Gpt::before_forward(int batch_size, int prompt_len, int steps) {
  if (steps == 0) {
    _launch_gpt_emb_layer->before_forward(batch_size,
                                          prompt_len, 0);
    for (auto iter : _gpt_layers_vec) {
      iter->before_forward(batch_size * tw_._beam_size, prompt_len, 0);
    }
    _lyr_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
    _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
    _generator_layer->before_forward(batch_size, prompt_len, 0);
  } else {
    _launch_gpt_emb_layer->before_forward(batch_size, 1,
                                          prompt_len + steps - 1);
    for (auto iter : _gpt_layers_vec) {
      iter->before_forward(batch_size * tw_._beam_size, 1,
                           prompt_len + steps - 1);
    }
    _lyr_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
    _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
    _generator_layer->before_forward(batch_size, prompt_len, steps);
  }
}

void Gpt::Infer() {
  int batch_size = input_shapes_[0][0], prompt_len = input_shapes_[0][1];

  /* --- notice that the order of forward should be the same with network --- */

#ifdef LIGHTSEQ_cuda
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int beam_idx = 0; beam_idx < tw_._beam_size; beam_idx++) {
      CHECK_GPU_ERROR(cudaMemcpyAsync(
          _inp_tokens->value<int>() +
              (batch_idx * tw_._beam_size + beam_idx) * tw_._max_step,
          _input_ptr + batch_idx * prompt_len, prompt_len * sizeof(int),
          cudaMemcpyDefault, _context_ptr->get_stream()));
    }
  }
#endif

  int steps = 0;
  while (steps + prompt_len < tw_._max_step) {
    before_forward(batch_size, prompt_len, steps);

    _launch_gpt_emb_layer->forward();
    for (auto iter : _gpt_layers_vec) {
      iter->forward();
    }

    if (steps == 0) {
      OpType_ *linear_inp_ptr = _lyr_norm_layer->input(0)->value<OpType_>();
      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int i = 0; i < tw_._beam_size; i++) {
          cudaMemcpyAsync(
              linear_inp_ptr +
                  (batch_idx * tw_._beam_size + i) * tw_._hidden_size,
              linear_inp_ptr + (batch_idx * tw_._beam_size * prompt_len +
                                i * prompt_len + prompt_len - 1) *
                                   tw_._hidden_size,
              tw_._hidden_size * sizeof(OpType_), cudaMemcpyDefault,
              _context_ptr->get_stream());
        }
      }
    }
    _lyr_norm_layer->forward();
    _linear_layer->forward();

    _generator_layer->forward();

    if (_generator_layer->is_stop()) {
      break;
    }
    if (_generate_method == GenerateMethod::BeamSearch) {
      _generator_layer->refresh_cache(_total_caches_k, _total_caches_v);
      if (steps + prompt_len + 1 < tw_._max_step) {
        Variable::swap_tensor(_inp_tokens, _out_tokens);
      }
    }
    steps++;
  }

  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int beam_idx = 0; beam_idx < tw_._beam_size; beam_idx++) {
      int *tmp_out_ptr = (_generate_method == GenerateMethod::BeamSearch)
                             ? _out_tokens->value<int>()
                             : _inp_tokens->value<int>();
      cudaMemcpyAsync(
          _gpt_out_ptr +
              (batch_idx * tw_._beam_size + beam_idx) * (steps + prompt_len),
          tmp_out_ptr + (batch_idx * tw_._beam_size + beam_idx) * tw_._max_step,
          (steps + prompt_len) * sizeof(int), cudaMemcpyDefault,
          _context_ptr->get_stream());
    }
  }
  cudaMemcpyAsync(_gpt_scores_ptr, _out_scores->value<float>(),
                  batch_size * tw_._beam_size * sizeof(float), cudaMemcpyDefault,
                  _context_ptr->get_stream());

  _context_ptr->synchronize();
  if (_generate_method == GenerateMethod::BeamSearch) {
    set_output_shape(0, {batch_size, tw_._beam_size, prompt_len + steps});
    set_output_shape(1, {batch_size, tw_._beam_size});
  } else {
    set_output_shape(0, {batch_size, 1, prompt_len + steps});
    set_output_shape(1, {batch_size, 1});
  }
}

void Gpt::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      _input_ptr = (int *)input_ptr;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Gpt::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      _gpt_out_ptr = (int *)output_ptr;
      break;

    case 1:
      _gpt_scores_ptr = (float *)output_ptr;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Gpt::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(_gpt_out_ptr);

    case 1:
      return static_cast<void *>(_gpt_scores_ptr);

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Gpt::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> Gpt::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._beam_size, tw_._max_step};

    case 1:
      return {_max_batch_size, tw_._beam_size};
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Gpt::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Gpt::get_output_dtype(int index) {
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
}  // namespace cuda
}  // namespace lightseq
