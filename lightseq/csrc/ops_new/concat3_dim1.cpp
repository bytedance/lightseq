#include "concat3_dim1.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Concat3Dim1<T1, T2>::operator()(Variable* inp,
                                                Variable* cache) {
#ifdef MODEL_INFER
  Variable* new_cache = new Variable("Concat3Dim1_new", cache);
#else
  Variable* new_cache = new Variable("Concat3Dim1_new");
#endif
  set_parents({inp, cache});
  this->set_children({new_cache});
  return new_cache;
}

template <typename T1, typename T2>
void Concat3Dim1<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* cache_ptr = (T1*)parent(1)->value();
  T1* real_val = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }
  
  if (_is_skip) {
    CHECK_GPU_ERROR(
        cudaMemcpyAsync((void*)real_val, (void*)inp_ptr,
                        _batchs * _hidden_size * _seq_len * sizeof(T1),
                        cudaMemcpyDefault, _stream));
    return;
  }

#ifdef MODEL_INFER
  launch_filling_concat3_dim1(cache_ptr, inp_ptr, _batchs * _heads, _max_steps, _hidden_size / _heads, _steps, 1, _stream);
#else
  launch_concat3_dim1(cache_ptr, inp_ptr, real_val, _batchs * _heads,
                      _hidden_size / _heads, _steps, 1, _stream);
#endif
}

template <typename T1, typename T2>
void Concat3Dim1<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* val_grad = (T1*)child(0)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef MODEL_INFER
  printf("Model infer does not have backward() function\n");
#else
  if (inp_grad != val_grad) {
    CHECK_GPU_ERROR(
        cudaMemcpyAsync((void*)inp_grad, (void*)val_grad,
                        _batchs * _hidden_size * _seq_len * sizeof(T2),
                        cudaMemcpyDefault, _stream));
  }
#endif
}

template class Concat3Dim1<float, float>;
template class Concat3Dim1<__half, __half>;

}  // namespace lightseq
