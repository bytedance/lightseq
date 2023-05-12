#include "llama.h"

namespace lightseq {
Llama::Llama(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"llama_out"}), _max_batch_size(max_batch_size) {
  printf("*** model max_batch_size: %d ***\n", max_batch_size);
  /* --- step.1 initial context --- */
  Context::create_global_context(StatusType::Inference);
  _context_ptr = Context::global_instance();

  /* --- step.2 load model weights into GPU memory --- */
  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path, _gen_conf);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }
  _generate_method = get_generate_method(tw_._generate_method);
  if (_generate_method != GenerateMethod::BeamSearch) {
    tw_._beam_size = 1;
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  _inp_tokens = new Variable("inp_tokens", g_dtype<int>());

  /* --- step.4 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;
  _launch_llama_emb_layer.reset(new LaunchLlamaEmbLayer<OpType_>(
      max_batch_tokens, tw_._max_step, _max_batch_size, tw_._beam_size,
      tw_._hidden_size));
  _launch_llama_emb_layer->load_params(tw_.get_src_emb_wei(), 0);

  int enc_wei_offset = 0;
  for (int idx = 0; idx < tw_._layer_num; idx++) {
    LlamaLayerPtr<OpType_, OpType_> llama_layer(
        new LlamaLayer<OpType_, OpType_>(max_batch_size, tw_._max_step,
                                         tw_._hidden_size, tw_._inner_size,
                                         tw_._head_num, tw_._beam_size));
    enc_wei_offset +=
        llama_layer->load_params(tw_.get_enc_wei(), enc_wei_offset);
    _llama_layer_vec.push_back(llama_layer);
  }

  _rms_norm_layer.reset(
      new RMSNormLayer<OpType_, OpType_>(max_batch_tokens, tw_._hidden_size));
  _rms_norm_layer->load_params(tw_.get_src_emb_wei(), 1);

  // intial Project hidden states to vocab logits
  _linear_layer.reset(new LinearLayer<OpType_, OpType_>(
      max_batch_size * tw_._beam_size, tw_._hidden_size, tw_._src_vocab_size,
      MATRIX_OP::NonTranspose, MATRIX_OP::NonTranspose, 1.f));
  _linear_layer->load_params(tw_.get_src_emb_wei(), 2);

  _generator_layer.reset(new GeneratorLayer<OpType_>(
      _generate_method, tw_._layer_num, max_batch_size, tw_._max_step,
      tw_._src_vocab_size, tw_._hidden_size, 1024, tw_._beam_size,
      tw_._diverse_lambda, tw_._dim_per_head, tw_._head_num,
      tw_._length_penalty, false));

  /* --- step.5 construct network --- */
  size_t cache_size = max_batch_tokens * tw_._beam_size * tw_._hidden_size;
  _total_caches_k = new Variable("total_caches_k", cache_size * tw_._layer_num,
                                 g_dtype<OpType_>(), DataType::kNotSupported,
                                 VariableType::RegressiveVariable);
  _total_caches_v = new Variable("total_caches_v", cache_size * tw_._layer_num,
                                 g_dtype<OpType_>(), DataType::kNotSupported,
                                 VariableType::RegressiveVariable);

  // note regress begin
  _context_ptr->regress_begin();

  std::tuple<Variable *, Variable *, Variable *> llama_emb_outs =
      (*_launch_llama_emb_layer)(_inp_tokens);
  Variable *llama_emb = std::get<0>(llama_emb_outs);
  Variable *pad_mask = std::get<1>(llama_emb_outs);
  pad_mask->set_regress_var();
  size_t cache_offset = 0;
  for (auto iter : _llama_layer_vec) {
    Variable *cache_k = new Variable("cache_k", _total_caches_k);
    cache_k->set_offset(cache_offset, {cache_size});
    Variable *cache_v = new Variable("cache_v", _total_caches_v);
    cache_v->set_offset(cache_offset, {cache_size});
    llama_emb = (*iter)(llama_emb, cache_k, cache_v, pad_mask);
    cache_offset += cache_size;
  }
  llama_emb = (*_rms_norm_layer)(llama_emb);
  Variable *logits_prob = (*_linear_layer)(llama_emb);

  std::tuple<Variable *, Variable *> gen_outs =
      (*_generator_layer)(logits_prob, _inp_tokens);

  // note regress_end
  _context_ptr->regress_end();

  _out_tokens = std::get<0>(gen_outs);
  _inp_tokens->malloc_memory(max_batch_size * tw_._beam_size * tw_._max_step);
  _out_tokens->malloc_memory(max_batch_size * tw_._beam_size * tw_._max_step);

  _context_ptr->build();
  printf("Finish construct network!\n");
}

Llama::~Llama() {}

void Llama::before_forward(int batch_size, int prompt_len, int steps) {
  if (steps == 0) {
    _launch_llama_emb_layer->before_forward(batch_size, prompt_len, 0);
    for (auto iter : _llama_layer_vec) {
      iter->before_forward(batch_size * tw_._beam_size, prompt_len, 0);
    }
    _rms_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
    _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
    _generator_layer->before_forward(batch_size, prompt_len, 0);
  } else {
    _launch_llama_emb_layer->before_forward(batch_size, 1,
                                            prompt_len + steps - 1);
    for (auto iter : _llama_layer_vec) {
      iter->before_forward(batch_size * tw_._beam_size, 1,
                           prompt_len + steps - 1);
    }
    _rms_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
    _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
    _generator_layer->before_forward(batch_size, prompt_len, steps);
  }
}

void Llama::Infer() {
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
  while (steps + prompt_len < tw_._max_step && steps < _gen_conf->_max_new_tokens) {
    before_forward(batch_size, prompt_len, steps);

    _launch_llama_emb_layer->forward();
    for (auto iter : _llama_layer_vec) {
      iter->forward();
    }

    if (steps == 0) {
      OpType_ *linear_inp_ptr = _rms_norm_layer->input(0)->value<OpType_>();
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
    _rms_norm_layer->forward();
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
      int tmp_idx = batch_idx * tw_._beam_size + beam_idx;
      int *tmp_out_ptr = (_generate_method == GenerateMethod::BeamSearch)
                             ? _out_tokens->value<int>()
                             : _inp_tokens->value<int>();
      cudaMemcpyAsync(
          _llama_out_ptr + tmp_idx * (steps + prompt_len),
          tmp_out_ptr + tmp_idx * tw_._max_step,
          (steps + prompt_len) * sizeof(int), cudaMemcpyDefault,
          _context_ptr->get_stream());
    }
  }

  _context_ptr->synchronize();
  set_output_shape(0, {batch_size, tw_._beam_size, prompt_len + steps});
}

void Llama::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      _input_ptr = (int *)input_ptr;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Llama::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      _llama_out_ptr = (int *)output_ptr;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Llama::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(_llama_out_ptr);

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Llama::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

std::vector<int> Llama::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._beam_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Llama::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Llama::get_output_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}
}  // namespace lightseq
