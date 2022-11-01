#include "beam_search_topk.h"

namespace lightseq {

template <typename T>
std::tuple<Variable*, Variable*, Variable*> BeamSearchTopkOp<T>::operator()(
    Variable* logits, Variable* logit_bias, Variable* seq_probs,
    Variable* seq_score, Variable* alive_seq) {
  Variable* can_idx = new Variable(
      "can_idx", _max_batch_size * _beam_size * _trg_vocab_size * sizeof(int));

  Variable* can_score =
      new Variable("can_score", _max_batch_size * _beam_size * _trg_vocab_size *
                                    sizeof(float));

  Variable* num_beam_can = new Variable(
      "num_beam_can", (_max_batch_size * _beam_size + 1) * sizeof(int));

  this->set_parents({logits, logit_bias, seq_probs, seq_score, alive_seq});
  this->set_children({can_idx, can_score, num_beam_can});
  return std::make_tuple(can_idx, can_score, num_beam_can);
}

template <typename T>
void BeamSearchTopkOp<T>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();
  T* logits_ptr = (T*)parent(0)->value();
  T* logits_bias_ptr = (T*)parent(1)->value();
  float* seq_probs_ptr = (float*)parent(2)->value();
  float* seq_score_ptr = (float*)parent(3)->value();
  int* alive_seq_ptr = (int*)parent(4)->value();

  int* can_idx_ptr = (int*)child(0)->value();
  float* can_score_ptr = (float*)child(1)->value();
  int* num_beam_can_ptr = (int*)child(2)->value();

  /*
    step 1. logits bias and softmax,
      select rough topk candidate for every batch item,
      record the candidate's beam_id, vocab_id and probability
  */

  cudaMemsetAsync(num_beam_can_ptr, 0, sizeof(int), stream);

  select_beam_rough_topk_launcher(
      logits_ptr, logits_bias_ptr, seq_probs_ptr, seq_score_ptr, alive_seq_ptr,
      can_idx_ptr, can_score_ptr, num_beam_can_ptr, _trg_vocab_size, _max_step,
      _length_norm, _cur_step, _step_token_num, _max_thread_per_block, stream,
      _beam_size, _diverse_lambda, _end_id);

  thrust::exclusive_scan(thrust::cuda::par.on(stream), num_beam_can_ptr + 1,
                         num_beam_can_ptr + 1 + _step_token_num,
                         num_beam_can_ptr + 1);

  /* ---step 2. sort the candidate with their probability--- */
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_host_can_num_batch, num_beam_can_ptr,
                                  sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream));

  if (_diverse_lambda != 0) {
    if (_host_can_num_batch < _cub_sort_buffer_bytes / 160) {
      CHECK_GPU_ERROR(cub::DeviceRadixSort::SortPairsDescending(
          (void*)logits_ptr, _cub_sort_buffer_bytes, can_score_ptr,
          can_score_ptr, can_idx_ptr, can_idx_ptr, _host_can_num_batch, 0,
          sizeof(float) * 8, stream));
    } else {
      thrust::sort_by_key(thrust::cuda::par.on(stream), can_score_ptr,
                          can_score_ptr + _host_can_num_batch, can_idx_ptr,
                          thrust::greater<float>());
    }
    ker_diverse_beam_search_launcher(can_score_ptr, can_idx_ptr,
                                     num_beam_can_ptr, _step_token_num,
                                     _max_thread_per_block, stream, _beam_size,
                                     _diverse_lambda, _trg_vocab_size);
  }

  thrust::sort_by_key(thrust::cuda::par.on(stream), can_score_ptr,
                      can_score_ptr + _host_can_num_batch, can_idx_ptr,
                      thrust::greater<float>());
}

template class BeamSearchTopkOp<float>;
template class BeamSearchTopkOp<__half>;

}  // namespace lightseq
