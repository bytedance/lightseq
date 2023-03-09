#include "sampling.h"

namespace lightseq {

template <typename T>
SamplingOp<T>::SamplingOp(GenerateMethod gm, int max_batch_size, int max_step,
                          int max_thread_per_block, int trg_vocab_size,
                          int topk, float topp, int eos_id)
    : Operator("SamplingOp"),
      _generate_method(gm),
      _max_batch_size(max_batch_size),
      _max_step(max_step),
      _max_thread_per_block(max_thread_per_block),
      _trg_vocab_size(trg_vocab_size),
      _topk(topk),
      _topp(topp),
      _eos_id(eos_id) {
#ifdef LIGHTSEQ_cuda
  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_curandstate,
                             _max_batch_size * sizeof(curandState)));
  cudaStream_t _stream = _context_ptr->get_stream();
  cuda::ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);
  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_unfinished, sizeof(int)));
#endif
}

template <typename T>
Variable* SamplingOp<T>::operator()(Variable* logits, Variable* logit_bias,
                                    Variable* token_ids) {
  set_parents({logits, logit_bias, token_ids});

  _out_token_ids = new Variable("out_token_ids", _max_batch_size * _max_step,
                                g_dtype<int>());

  set_children({_out_token_ids});
  return _out_token_ids;
}

template <typename T>
void SamplingOp<T>::forward() {
  T* logits_ptr = parent(0)->value<T>();
  T* logits_bias_ptr = parent(1)->value<T>();
  int* inp_tokens_ptr = parent(2)->value<int>();

  int* out_tokens_ptr = _out_token_ids->value<int>();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_unfinished, 0, sizeof(int), _stream));
  if (_generate_method == GenerateMethod::Topk) {
    cuda::ker_topk_sample_launcher<T>(
        _batch_size, _seq_len, _max_step, _logits_seq_len,
        _max_thread_per_block, _stream, logits_ptr, logits_bias_ptr,
        inp_tokens_ptr, out_tokens_ptr, _trg_vocab_size, _topk, _p_d_unfinished,
        _p_d_curandstate, _eos_id);
  } else if (_generate_method == GenerateMethod::Topp) {
    cuda::ker_topp_sample_launcher<T>(
        _batch_size, _seq_len, _max_step, _logits_seq_len,
        _max_thread_per_block, _stream, logits_ptr, logits_bias_ptr,
        inp_tokens_ptr, out_tokens_ptr, _trg_vocab_size, _topp, _p_d_unfinished,
        _p_d_curandstate, _eos_id);
  }

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_unfinished, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
#endif
}

template class SamplingOp<float>;
#ifdef LIGHTSEQ_cuda
template class SamplingOp<__half>;
#endif

}  // namespace lightseq
