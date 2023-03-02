#include "concat3_dim1.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Concat3Dim1<T1, T2>::operator()(Variable* inp, Variable* cache) {
  if (!_is_continuous_cache) {
    _new_cache = new Variable("cache_out", cache);
    _new_cache->set_offset(0, {_mx_sz0, _mx_sz1, _mx_sz2});
  } else {
    _new_cache = new Variable("cache_out", g_dtype<T1>(),
                              g_dtype<T2>());  // just for pybind interface
  }

  set_parents({inp, cache});
  this->set_children({_new_cache});
  return _new_cache;
}

template <typename T1, typename T2>
void Concat3Dim1<T1, T2>::forward() {
  T1* inp_ptr = (T1*)parent(0)->value();
  T1* cache_ptr = (T1*)parent(1)->value();
  T1* real_val = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  if (_is_skip) {
    CHECK_GPU_ERROR(cudaMemcpyAsync((void*)real_val, (void*)inp_ptr,
                                    _sz0 * _sz1_1 * _mx_sz2 * sizeof(T1),
                                    cudaMemcpyDefault, _stream));
    return;
  }

  if (!_is_continuous_cache) {
    cuda::launch_filling_concat3_dim1(cache_ptr, inp_ptr, _sz0, _mx_sz1,
                                      _mx_sz2, _sz1_0, _sz1_1, _stream);
  } else {
    cuda::launch_concat3_dim1(cache_ptr, inp_ptr, real_val, _sz0, _mx_sz2,
                              _sz1_0, _sz1_1, _stream);
  }
#endif
}

template <typename T1, typename T2>
void Concat3Dim1<T1, T2>::backward() {
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* val_grad = (T1*)child(0)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  if (!_is_continuous_cache)
    printf("Model infer does not have backward() function\n");
  else {
    if (inp_grad != val_grad) {
      CHECK_GPU_ERROR(cudaMemcpyAsync((void*)inp_grad, (void*)val_grad,
                                      _sz0 * _sz1_1 * _mx_sz2 * sizeof(T2),
                                      cudaMemcpyDefault, _stream));
    }
  }
#endif
}

template class Concat3Dim1<float, float>;
#ifdef LIGHTSEQ_cuda
template class Concat3Dim1<__half, __half>;
#endif

}  // namespace lightseq
