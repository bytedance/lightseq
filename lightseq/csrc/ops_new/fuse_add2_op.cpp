#include "fuse_add2_op.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* FuseAdd2Op<T1, T2>::operator()(Variable* inpA, Variable* inpB) {
  _result = new Variable("FuseAdd2Op_out", _max_batch_tokens * _hidden_dim,
                         g_dtype<T1>(), g_dtype<T2>());
  set_parents({inpA, inpB});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void FuseAdd2Op<T1, T2>::forward() {

  T1* inpA_ptr = (T1*)parent(0)->value();
  T1* inpB_ptr = (T1*)parent(1)->value();
  T1* out_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
    cudaStream_t stream = _context_ptr->get_stream();
    cuda::launch_fused_add2(out_ptr, inpA_ptr, inpB_ptr, _batch_size, _seq_len, _hidden_dim, stream);
#endif 
}


template class FuseAdd2Op<float, float>;
#ifdef LIGHTSEQ_cuda
template class FuseAdd2Op<__half, __half>;
#endif
}  // namespace lightseq
