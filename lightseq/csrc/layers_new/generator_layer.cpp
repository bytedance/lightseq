#include "generator_layer.h"

namespace lightseq {

template <typename T>
GeneratorLayer<T>::GeneratorLayer(GenerateMethod gm, int nshared_dec_layer,
                                  int max_batch_size, int max_step,
                                  int trg_vocab_size, int hidden_size,
                                  int max_thread_per_block, int beam_size,
                                  float diverse_lambda, int dim_per_head,
                                  int end_id, int head_num,
                                  float length_penalty, int topk, float topp,
                                  bool has_logits_bias)
    : Layer("GeneratorLayer"),
      _generate_method(gm),
      _trg_vocab_size(trg_vocab_size),
      _has_logits_bias(has_logits_bias) {
  if (_generate_method == GenerateMethod::BeamSearch) {
    _beam_search = new BeamSearchTopOp<T>(
        nshared_dec_layer, max_batch_size, max_step, trg_vocab_size,
        hidden_size, max_thread_per_block, beam_size, diverse_lambda,
        dim_per_head, end_id, head_num, length_penalty);
  } else {
    _sampling =
        new SamplingOp<T>(gm, max_batch_size, max_step, max_thread_per_block,
                          trg_vocab_size, topk, topp, end_id);
  }

  _logit_bias = new Variable("logits_bias", g_dtype<T>());

  if (!has_logits_bias) {
    auto allocator_ptr = _context_ptr->allocator();
    char* tmp_logit_bias_ptr =
        allocator_ptr->malloc_mem(trg_vocab_size * sizeof(T));
#ifdef LIGHTSEQ_cuda
    CHECK_GPU_ERROR(
        cudaMemset(tmp_logit_bias_ptr, T(0.), trg_vocab_size * sizeof(T)));
#else
    memset(tmp_logit_bias_ptr, T(0.), trg_vocab_size * sizeof(T));
#endif
    _logit_bias->set_value(tmp_logit_bias_ptr);
  }

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T>
std::tuple<Variable*, Variable*> GeneratorLayer<T>::operator()(
    Variable* logits, Variable* alive_seq) {
  set_inputs({logits, alive_seq});

  Variable* alive_seq_out = nullptr;
  Variable* seq_score = nullptr;

  if (GenerateMethod::BeamSearch == _generate_method) {
    std::tuple<Variable*, Variable*> beam_search_outs =
        (*_beam_search)(logits, _logit_bias, alive_seq);
    alive_seq_out = std::get<0>(beam_search_outs);
    seq_score = std::get<1>(beam_search_outs);
  } else {
    std::tuple<Variable*, Variable*> sample_outs = (*_sampling)(logits, _logit_bias, alive_seq);
    alive_seq_out = std::get<0>(sample_outs);
    seq_score = std::get<1>(sample_outs);
  }

  set_outputs({alive_seq_out, seq_score});
  return std::make_tuple(alive_seq_out, seq_score);
}

template <typename T>
void GeneratorLayer<T>::before_forward(int batch_size, int prompt_len,
                                       int cur_step) {
  if (_generate_method == GenerateMethod::BeamSearch) {
    _beam_search->before_forward(batch_size, prompt_len, cur_step);
  } else {
    _sampling->before_forward(batch_size, prompt_len, cur_step, 1);
  }
}

template <typename T>
int GeneratorLayer<T>::load_params(const std::vector<const T*>& para_vec,
                                   int offset) {  // for inference
  int size = 0;
  if (_has_logits_bias) {
    _logit_bias->set_value((char*)para_vec[offset + size]), size++;
    _logit_bias->set_shape({_trg_vocab_size});
  }
  return size;
}

template <typename T>
bool GeneratorLayer<T>::is_stop() {
  switch (_generate_method) {
    case GenerateMethod::BeamSearch:
      return _beam_search->is_stop();
    case GenerateMethod::Topk:
      return _sampling->is_stop();
    case GenerateMethod::Topp:
      return _sampling->is_stop();
  }
  return true;
}

template <typename T>
void GeneratorLayer<T>::refresh_cache(Variable* caches_k, Variable* caches_v) {
  if (_generate_method == GenerateMethod::BeamSearch) {
    _beam_search->refresh_cache(caches_k, caches_v);
  }
}

}  // namespace lightseq
