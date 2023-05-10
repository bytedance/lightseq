#pragma once
#include "declaration.h"
#include "node.h"
#include "cmath"

namespace lightseq {

template <typename T1, typename T2>
class RotaryPositionQk : public Operator {
 private:
  T1* _sin_ptr;
  T1* _cos_ptr;
  size_t _max_step;
  size_t _max_batch_size;
  size_t _batch_size;
  size_t _head_num;
  size_t _head_dim;
  size_t _offset_seq_len;
  size_t _query_len;

  T1* _device_sin_ptr;
  T1* _device_cos_ptr;

  Variable* _result;

 public:
  RotaryPositionQk(int max_batch_size, int max_step, int head_num, int head_dim)
      : Operator("RotaryPositionQk"),
        _max_batch_size(max_batch_size),
        _max_step(max_step),
        _head_num(head_num),
        _head_dim(head_dim) {
    if (head_dim & 1) {
      printf(
          "Error! head dim should be even number while using RotaryPositionQk "
          "Operator.\n");
      exit(0);
    }

    int total_size = max_step * head_dim / 2;
    _sin_ptr = (T1*)malloc(total_size * sizeof(T1));
    _cos_ptr = (T1*)malloc(total_size * sizeof(T1));

    for (int i = 0; i < head_dim / 2; i++) {
      float theta = std::pow(10000, -2. * i / head_dim);
      for (int j = 0; j < max_step; j++) {
        T1 sin_val, cos_val;
        if (std::is_same<T1, float>::value) {
          sin_val = sin(j * theta), cos_val = cos(j * theta);
        } else {
          sin_val = __float2half(sin(j * theta)),
          cos_val = __float2half(cos(j * theta));
        }
        *(_sin_ptr + j * head_dim / 2 + i) =
            sin_val;  // shape: [max_step, head_dim / 2]
        *(_cos_ptr + j * head_dim / 2 + i) =
            cos_val;  // shape: [max_step, head_dim / 2]
      }
    }

#ifdef LIGHTSEQ_cuda
    _device_sin_ptr =
        (T1*)_context_ptr->allocator()->malloc_mem(total_size * sizeof(T1));
    _device_cos_ptr =
        (T1*)_context_ptr->allocator()->malloc_mem(total_size * sizeof(T1));
    CHECK_GPU_ERROR(cudaMemcpy(_device_sin_ptr, _sin_ptr,
                               total_size * sizeof(T1), cudaMemcpyDefault));
    CHECK_GPU_ERROR(cudaMemcpy(_device_cos_ptr, _cos_ptr,
                               total_size * sizeof(T1), cudaMemcpyDefault));
    free(_sin_ptr);
    _sin_ptr = nullptr;
    free(_cos_ptr);
    _cos_ptr = nullptr;
#else
    _device_sin_ptr = _sin_ptr;
    _device_cos_ptr = _cos_ptr;
#endif
  }

  virtual ~RotaryPositionQk() {}

  void before_forward(int batch_size, int offset_seq_len, int query_len) {
    _batch_size = batch_size;
    _offset_seq_len = offset_seq_len;
    _query_len = query_len;
    _result->set_shape({_batch_size, _head_num, _query_len, _head_dim});
  }

  Variable* operator()(Variable* inp_tensor, Variable* cache_k,
                       Variable* cache_v);

  void forward() override;

  void backward() override {}
};

}  // namespace lightseq
