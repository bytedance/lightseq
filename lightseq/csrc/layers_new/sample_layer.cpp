#include "sample_layer.h"

namespace lightseq {

template <typename T>
SampleLayer<T>::SampleLayer(int nshared_dec_layer, int max_batch_size,
                            int max_step, int trg_vocab_size, int hidden_size,
                            int max_thread_per_block, int beam_size,
                            int diverse_lambda, int dim_per_head, int end_id,
                            int head_num, float length_penalty)
    : Layer("SampleLayer"),
      _beam_search(new BeamSearchTopOp<T>(
          nshared_dec_layer, max_batch_size, max_step, trg_vocab_size,
          hidden_size, max_thread_per_block, beam_size, diverse_lambda,
          dim_per_head, end_id, head_num, length_penalty)) {
  _logit_bias = new Variable("logits_bias", g_dtype<T>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T>
std::tuple<Variable*, Variable*> SampleLayer<T>::operator()(
    Variable* logits, Variable* alive_seq, Variable* total_cache_k,
    Variable* total_cache_v) {
  set_inputs({logits, alive_seq, total_cache_k, total_cache_v});

  std::tuple<Variable*, Variable*> beam_search_outs = (*_beam_search)(
      logits, _logit_bias, alive_seq, total_cache_k, total_cache_v);
  Variable* alive_seq_out = std::get<0>(beam_search_outs);
  Variable* seq_score = std::get<1>(beam_search_outs);

  set_outputs({alive_seq_out, seq_score});
  return beam_search_outs;
}

template <typename T>
void SampleLayer<T>::before_forward(int batch_size, int cur_step) {
  _beam_search->before_forward(batch_size, cur_step);
}

template <typename T>
int SampleLayer<T>::load_params(const std::vector<const T*>& para_vec,
                                int offset) {  // for inference
  int size = 0;
  _logit_bias->set_value((char*)para_vec[offset + size]), size++;

  return size;
}

}  // namespace lightseq
