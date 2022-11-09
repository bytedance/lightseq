#include "beam_search_topk.h"

namespace lightseq {

template <typename T>

BeamSearchTopOp<T>::BeamSearchTopOp(int nshared_dec_layer, int max_batch_size,
                                    int max_step, int trg_vocab_size,
                                    int hidden_size, int max_thread_per_block,
                                    int beam_size, int diverse_lambda,
                                    int dim_per_head, int end_id, int head_num)
    : Operator("BeamSearchTopOp"),
      _nshared_dec_layer(nshared_dec_layer),
      _max_batch_size(max_batch_size),
      _max_step(max_step),
      _trg_vocab_size(trg_vocab_size),
      _max_thread_per_block(max_thread_per_block),
      _beam_size(beam_size),
      _diverse_lambda(diverse_lambda),
      _dim_per_head(dim_per_head),
      _end_id(end_id),
      _head_num(head_num),
      _cub_sort_buffer_bytes(max_batch_size * beam_size * trg_vocab_size *
                             sizeof(T)),
      _cache_size(max_batch_size * max_step * hidden_size * _beam_size) {}

template <typename T>
std::tuple<Variable*, Variable*, Variable*, Variable*>
BeamSearchTopOp<T>::operator()(Variable* logits, Variable* logit_bias,
                               Variable* seq_probs, Variable* seq_score,
                               Variable* alive_seq, Variable* caches_k,
                               Variable* caches_k_buf, Variable* caches_v,
                               Variable* caches_v_buf) {
  set_parents({logits, logit_bias, seq_probs, seq_score, alive_seq, caches_k,
               caches_k_buf, caches_v, caches_v_buf});

  Variable* can_idx = new Variable(
      "can_idx", _max_batch_size * _beam_size * _trg_vocab_size * sizeof(int));
  Variable* can_score =
      new Variable("can_score", _max_batch_size * _beam_size * _trg_vocab_size *
                                    sizeof(float));
  Variable* num_beam_can = new Variable(
      "num_beam_can", (_max_batch_size * _beam_size + 1) * sizeof(int));

  Variable* alive_seq_out = new Variable(
      "alive_seq_out", _max_batch_size * _beam_size * _max_step * sizeof(int));

  set_children({can_idx, can_score, num_beam_can, alive_seq_out});
  return std::make_tuple(can_idx, can_score, num_beam_can, alive_seq_out);
}

template <typename T>
void BeamSearchTopOp<T>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();
  T* logits_ptr = (T*)parent(0)->value();
  T* logits_bias_ptr = (T*)parent(1)->value();
  float* seq_probs_ptr = (float*)parent(2)->value();
  float* seq_score_ptr = (float*)parent(3)->value();
  int* alive_seq_ptr = (int*)parent(4)->value();
  Variable* caches_k = parent(5);
  Variable* caches_k_buf = parent(6);
  Variable* caches_v = parent(7);
  Variable* caches_v_buf = parent(8);

  int* can_idx_ptr = (int*)child(0)->value();
  float* can_score_ptr = (float*)child(1)->value();
  int* num_beam_can_ptr = (int*)child(2)->value();
  int* alive_seq_out = (int*)child(3)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

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

  /*
    step 3. refresh alive_seq, seq_probs, seq_score, num_finish_beam
      based on sorted candidate.
      Deciding whether early stop based on num_finish_beam
  */
  CHECK_GPU_ERROR(cudaMemsetAsync(num_beam_can_ptr, 0, sizeof(int), stream));
  ker_refresh_result<<<dim3(_batch_size, _beam_size), _max_step, 0, stream>>>(
      can_idx_ptr, can_score_ptr, num_beam_can_ptr + 1, alive_seq_ptr,
      alive_seq_out, seq_probs_ptr, seq_score_ptr, num_beam_can_ptr,
      _trg_vocab_size, _cur_step, _length_norm, _diverse_lambda, _end_id);
  // swap alive_seq
  Variable::swap_tensor(parent(4), child(3));
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_host_can_num_batch, num_beam_can_ptr,
                                  sizeof(int), cudaMemcpyDefault, stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream));

  if (_host_can_num_batch == _step_token_num) {
    return ;
  }

  /* ---step 4. refresh cache: k, v for decoder self attention--- */
  if (_cur_step > 0) {
    ker_refresh_cache_launcher<T>(
        _nshared_dec_layer * (_cur_step + 1), _step_token_num * 2,
        _max_thread_per_block, stream, num_beam_can_ptr + 1, can_idx_ptr,
        (T*)caches_k->value(), (T*)caches_v->value(), (T*)caches_k_buf->value(),
        (T*)caches_v_buf->value(), _cache_size, _beam_size, _dim_per_head,
        _head_num, _trg_vocab_size, _cur_step, _max_step,
        _diverse_lambda != 0, _end_id);
    Variable::swap_tensor(caches_k, caches_k_buf);
    Variable::swap_tensor(caches_v, caches_v_buf);
  }
}

template class BeamSearchTopOp<float>;
template class BeamSearchTopOp<__half>;

}  // namespace lightseq
