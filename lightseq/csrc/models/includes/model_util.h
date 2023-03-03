#pragma once
#include "layer.h"

namespace lightseq {

GenerateMethod get_generate_method(std::string method_) {
  if (method_ == "topk") return GenerateMethod::Topk;
  if (method_ == "topp") return GenerateMethod::Topp;
  if (method_ == "beam_search") return GenerateMethod::BeamSearch;

  printf("Error!\n");
  return GenerateMethod::UnDefined;
}

template <typename T>
void refresh_cache(int grid_dim_x, int grid_dim_y, int block_dim,
                   cudaStream_t stream, const int* num_can_per_beam,
                   const int* can_idx, Variable* caches_k, Variable* caches_v,
                   Variable* caches_k_buf, Variable* caches_v_buf,
                   size_t cache_block_size, int beam_size, int dim_per_head,
                   int head_num, int vocab_size, int cur_step, int max_step,
                   bool diverse, int end_id) {
  if (cur_step <= 0) {
    return;
  }

  T* caches_k_ptr = caches_k->value<T>();
  T* caches_v_ptr = caches_v->value<T>();
  T* caches_k_buf_ptr = caches_k_buf->value<T>();
  T* caches_v_buf_ptr = caches_v_buf->value<T>();

  cuda::ker_refresh_cache_launcher<T>(
      grid_dim_x, grid_dim_y, block_dim, stream, num_can_per_beam + 1, can_idx,
      (T*)caches_k_ptr, (T*)caches_v_ptr, (T*)caches_k_buf_ptr,
      (T*)caches_v_buf_ptr, cache_block_size, beam_size, dim_per_head, head_num,
      vocab_size, cur_step, max_step, diverse, end_id);
  Variable::swap_tensor(caches_k, caches_k_buf);
  Variable::swap_tensor(caches_v, caches_v_buf);
}

}  // namespace lightseq
