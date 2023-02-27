#include "beam_search_topk.h"

namespace lightseq {

float host_length_norm_func(int length, float alpha) {
  if (alpha < 0.f) return 1.f / length;
  return std::pow((5.f + length) / 6.f, -alpha);
}

template <typename T>
BeamSearchTopOp<T>::BeamSearchTopOp(size_t nshared_dec_layer,
                                    size_t max_batch_size, size_t max_step,
                                    size_t trg_vocab_size, size_t hidden_size,
                                    size_t max_thread_per_block,
                                    size_t beam_size, size_t diverse_lambda,
                                    size_t dim_per_head, int end_id,
                                    size_t head_num, float length_penalty)
    : Operator("BeamSearchTopOp"),
      _nshared_dec_layer(nshared_dec_layer),
      _max_batch_size(max_batch_size),
      _max_step(max_step),
      _trg_vocab_size(trg_vocab_size),
      _hidden_size(hidden_size),
      _max_thread_per_block(max_thread_per_block),
      _beam_size(beam_size),
      _diverse_lambda(diverse_lambda),
      _dim_per_head(dim_per_head),
      _end_id(end_id),
      _head_num(head_num),
      _cub_sort_buffer_bytes(max_batch_size * beam_size * trg_vocab_size *
                             sizeof(T)),
      _cache_size(max_batch_size * max_step * hidden_size * beam_size),
      _host_alive_seq_probs(max_batch_size * beam_size,
                            host_min_log_probability / 2),
      _host_length_norm(max_step, 1.f) {
  for (int i = 0; i < _host_alive_seq_probs.size(); i += beam_size) {
    _host_alive_seq_probs[i] = 0.f;
  }

  if (length_penalty >= 0) {
    for (int i = 0; i < _host_length_norm.size(); i++) {
      _host_length_norm[i] = host_length_norm_func(i + 1, length_penalty);
    }
  }
}

template <typename T>
std::tuple<Variable*, Variable*> BeamSearchTopOp<T>::operator()(
    Variable* logits, Variable* logit_bias, Variable* alive_seq,
    Variable* caches_k, Variable* caches_v) {
  set_parents({logits, logit_bias, alive_seq, caches_k, caches_v});

  _alive_seq_out =
      new Variable("alive_seq_out", _max_batch_size * _beam_size * _max_step,
                   g_dtype<int>());

  _seq_score = new Variable(
      "seq_score", _max_batch_size * _beam_size * _max_step, g_dtype<int>(),
      DataType::kNotSupported, VariableType::RegressiveVariable);

  // initial own variable

  _num_beam_can = new Variable("num_beam_can", _max_batch_size * _beam_size + 1,
                               g_dtype<int>(), DataType::kNotSupported,
                               VariableType::RegressiveVariable);
  _can_idx = new Variable(
      "can_idx", _max_batch_size * _beam_size * _trg_vocab_size, g_dtype<int>(),
      DataType::kNotSupported, VariableType::RegressiveVariable);
  _can_score =
      new Variable("can_score", _max_batch_size * _beam_size * _trg_vocab_size,
                   g_dtype<float>(), DataType::kNotSupported,
                   VariableType::RegressiveVariable);
  _seq_prob =
      new Variable("seq_prob", _max_batch_size * _beam_size, g_dtype<float>(),
                   DataType::kNotSupported, VariableType::RegressiveVariable);

  _caches_k_buf = new Variable("caches_k_buf",
                               _nshared_dec_layer * _max_batch_size *
                                   _max_step * _beam_size * _hidden_size,
                               g_dtype<T>(), DataType::kNotSupported,
                               VariableType::RegressiveVariable);
  _caches_v_buf = new Variable("caches_v_buf",
                               _nshared_dec_layer * _max_batch_size *
                                   _max_step * _beam_size * _hidden_size,
                               g_dtype<T>(), DataType::kNotSupported,
                               VariableType::RegressiveVariable);

  set_children({_alive_seq_out, _seq_score});
  return std::make_tuple(_alive_seq_out, _seq_score);
}

template <typename T>
void BeamSearchTopOp<T>::forward() {
  T* logits_ptr = (T*)parent(0)->value();
  T* logits_bias_ptr = (T*)parent(1)->value();
  int* alive_seq_ptr = (int*)parent(2)->value();
  Variable* caches_k = parent(3);
  T* caches_k_ptr = (T*)caches_k->value();
  Variable* caches_v = parent(4);
  T* caches_v_ptr = (T*)caches_v->value();
  T* caches_k_buf_ptr = (T*)_caches_k_buf->value();
  T* caches_v_buf_ptr = (T*)_caches_v_buf->value();

  int* alive_seq_out = (int*)child(0)->value();
  float* seq_score_ptr = (float*)child(1)->value();

  float* seq_probs_ptr = (float*)_seq_prob->value();
  int* can_idx_ptr = (int*)_can_idx->value();
  float* can_score_ptr = (float*)_can_score->value();
  int* num_beam_can_ptr = (int*)_num_beam_can->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  if (_cur_step == 0) {
    CHECK_GPU_ERROR(cudaMemcpyAsync(
        (void*)seq_probs_ptr, (void*)_host_alive_seq_probs.data(),
        sizeof(float) * _batch_size * _beam_size, cudaMemcpyDefault, stream));
  }

  /*
    step 1. logits bias and softmax,
      select rough topk candidate for every batch item,
      record the candidate's beam_id, vocab_id and probability
  */

  cudaMemsetAsync(num_beam_can_ptr, 0, sizeof(int), stream);

  cuda::select_beam_rough_topk_launcher(
      logits_ptr, logits_bias_ptr, seq_probs_ptr, seq_score_ptr, alive_seq_ptr,
      can_idx_ptr, can_score_ptr, num_beam_can_ptr, _trg_vocab_size, _max_step,
      _host_length_norm[_cur_step], _cur_step, _step_token_num,
      _max_thread_per_block, stream, _beam_size, _diverse_lambda, _end_id);

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
    cuda::ker_diverse_beam_search_launcher(
        can_score_ptr, can_idx_ptr, num_beam_can_ptr, _step_token_num,
        _max_thread_per_block, stream, _beam_size, _diverse_lambda,
        _trg_vocab_size);
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
  cuda::ker_refresh_result<<<dim3(_batch_size, _beam_size), _max_step, 0,
                             stream>>>(
      can_idx_ptr, can_score_ptr, num_beam_can_ptr + 1, alive_seq_ptr,
      alive_seq_out, seq_probs_ptr, seq_score_ptr, num_beam_can_ptr,
      _trg_vocab_size, _cur_step, _host_length_norm[_cur_step], _diverse_lambda,
      _end_id);

  // swap alive_seq
  // Variable::swap_tensor(parent(4), child(3));
  // don't swap alive_seq with alive_seq_buf in this function

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_host_can_num_batch, num_beam_can_ptr,
                                  sizeof(int), cudaMemcpyDefault, stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream));

#ifdef DEBUG_MODE
  for (int ii = 0; ii < _batch_size; ii++) {
    printf("++++++ _batch_size: %d ++++++\n", ii);
    for (int jj = 0; jj < _beam_size; jj++) {
      printf("++++++ _beam_size: %d ++++++\n", jj);
      print_vec(alive_seq_out + (ii * _beam_size + jj) * _max_step,
                "Batch token ids", _cur_step + 2);
      print_vec(seq_probs_ptr + ii * _beam_size + jj, "Batch probs", 1);
      print_vec(seq_score_ptr + ii * _beam_size + jj, "Batch scores", 1);
    }
  }
#endif

  if (_host_can_num_batch == _step_token_num) {
    return;
  }

  /* ---step 4. refresh cache: k, v for decoder self attention--- */
  if (_cur_step > 0) {
    cuda::ker_refresh_cache_launcher<T>(
        _nshared_dec_layer * (_cur_step + 1), _step_token_num * 2,
        _max_thread_per_block, stream, num_beam_can_ptr + 1, can_idx_ptr,
        (T*)caches_k_ptr, (T*)caches_v_ptr, (T*)caches_k_buf_ptr,
        (T*)caches_v_buf_ptr, _cache_size, _beam_size, _dim_per_head, _head_num,
        _trg_vocab_size, _cur_step, _max_step, _diverse_lambda != 0, _end_id);
    Variable::swap_tensor(caches_k, _caches_k_buf);
    Variable::swap_tensor(caches_v, _caches_v_buf);
  }
#endif
}

template class BeamSearchTopOp<float>;
#ifdef LIGHTSEQ_cuda
template class BeamSearchTopOp<__half>;
#endif

}  // namespace lightseq
