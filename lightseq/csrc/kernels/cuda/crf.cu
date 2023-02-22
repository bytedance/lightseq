#include "kernels.h"
#include "common.h"
#include "block_reduce.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {
static const int invalid_tag = -1;

__device__ void warp_reduce_max(cg::thread_block_tile<WARP_SIZE> g, float* val,
                                int* index) {
  float cur_max_v = *val;
  int cur_max_idx = *index;
  for (int i = 1; i < WARP_SIZE; i <<= 1) {
    float v = g.shfl_down(cur_max_v, i);
    int idx = g.shfl_down(cur_max_idx, i);
    if (v > cur_max_v) {
      cur_max_v = v;
      cur_max_idx = idx;
    }
  }  // note: only thread 0 will return result
  *val = cur_max_v;
  *index = cur_max_idx;
}

/**
@brief: ker_viterbi
Find the best tag sequence using Viterbi algorithm.
One thread block compute one sequence.

@thread
gridDim.x = batch_size
blockDim.x = WARP_SIZE
blockDim.y = WARP_SIZE

@param
start_transition: [num_tags]
end_transition: [num_tags]
transition: [num_tags, num_tags]
  transition[i, j] means the score of tag_j -> tag_i
emission: [batch_size, seq_len, num_tags]
mask: [batch_size, seq_len]
  0 for invalid token
bias: [num_tags]
best_score: [batch_size]
history: [batch_size, seq_len, num_tags]:
  i, j, k store the tag of i-th batch, j-th step when
  the tag of i-th batch, (j+1)-th step is k
best_tag: [batch_size, seq_len]
*/
template <typename T>
__global__ void ker_viterbi(const T* start_transition, const T* end_transition,
                            const T* transition, const T* emission,
                            const T* mask, const T* bias, float* best_score,
                            int* history, int* best_tags, int num_tags,
                            int seq_len) {
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  extern __shared__ float smen[];
  float* s_score = smen;
  float* s_next_score = smen + num_tags;

  // step 1. compute first step's score
  if (threadIdx.y == 0) {
    for (int cur_tag = threadIdx.x; cur_tag < num_tags; cur_tag += blockDim.x) {
      float linear_bias = bias ? float(bias[cur_tag]) : float(0);
      s_score[cur_tag] =
          float(
              emission[flat_3dim(blockIdx.x, 0, cur_tag, seq_len, num_tags)]) +
          linear_bias + float(start_transition[cur_tag]);
    }
  }
  b.sync();

  // step 2. compute last step's score
  int seq_idx;
  for (seq_idx = 1; seq_idx < seq_len; seq_idx++) {
    if (mask[flat_2dim(blockIdx.x, seq_idx, seq_len)] == CUDA_FLOAT_INF_NEG) {
      break;
    }
    for (int cur_tag = threadIdx.y; cur_tag < num_tags; cur_tag += blockDim.y) {
      float max_score = REDUCE_FLOAT_INF_NEG;
      int idx = 0;
      const T* cur_transition = transition + cur_tag * num_tags;
      for (int pre_tag = threadIdx.x; pre_tag < num_tags;
           pre_tag += blockDim.x) {
        float s = (float)s_score[pre_tag] + (float)cur_transition[pre_tag];
        if (s > max_score) {
          max_score = s;
          idx = pre_tag;
        }
      }  // col
      g.sync();
      warp_reduce_max(g, &max_score, &idx);
      if (threadIdx.x == 0) {
        float linear_bias = bias ? float(bias[cur_tag]) : float(0);
        s_next_score[cur_tag] =
            max_score +
            float(emission[flat_3dim(blockIdx.x, seq_idx, cur_tag, seq_len,
                                     num_tags)]) +
            linear_bias;
        history[flat_3dim(blockIdx.x, seq_idx - 1, cur_tag, seq_len,
                          num_tags)] = idx;
      }
    }  // row
    float* tmp = s_next_score;
    s_next_score = s_score;
    s_score = tmp;
    b.sync();
  }  // seq_len

  // step 3. compute last tag
  if (threadIdx.y != 0) {
    return;
  }
  float max_score = REDUCE_FLOAT_INF_NEG;
  int last_tag = 0;
  for (int cur_tag = threadIdx.x; cur_tag < num_tags; cur_tag += blockDim.x) {
    float s = (float)s_score[cur_tag] + (float)end_transition[cur_tag];
    if (s > max_score) {
      max_score = s;
      last_tag = cur_tag;
    }
  }
  g.sync();
  warp_reduce_max(g, &max_score, &last_tag);

  // step 4. compute full tag sequence
  if (threadIdx.x != 0) {
    return;
  }
  if (best_score) {
    best_score[blockIdx.x] = max_score;  // for debug
  }
  seq_idx--;
  best_tags[flat_2dim(blockIdx.x, seq_idx, seq_len)] = last_tag;
  for (int i = seq_idx - 1; i >= 0; i--) {
    last_tag = history[flat_3dim(blockIdx.x, i, last_tag, seq_len, num_tags)];
    best_tags[flat_2dim(blockIdx.x, i, seq_len)] = last_tag;
  }
  for (int i = seq_idx + 1; i < seq_len; i++) {
    best_tags[flat_2dim(blockIdx.x, i, seq_len)] = invalid_tag;
  }
}

template <>
__global__ void ker_viterbi<__half>(const __half* start_transition,
                                    const __half* end_transition,
                                    const __half* transition,
                                    const __half* emission, const __half* mask,
                                    const __half* bias, float* best_score,
                                    int* history, int* best_tags, int num_tags,
                                    int seq_len) {
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  extern __shared__ float smen[];
  float* s_score = smen;
  float* s_next_score = smen + num_tags;

  // step 1. compute first step's score
  if (threadIdx.y == 0) {
    for (int cur_tag = threadIdx.x; cur_tag < num_tags; cur_tag += blockDim.x) {
      float linear_bias = bias ? float(bias[cur_tag]) : float(0);
      s_score[cur_tag] =
          float(
              emission[flat_3dim(blockIdx.x, 0, cur_tag, seq_len, num_tags)]) +
          linear_bias + float(start_transition[cur_tag]);
    }
  }
  b.sync();

  // step 2. compute last step's score
  int seq_idx;
  for (seq_idx = 1; seq_idx < seq_len; seq_idx++) {
    if (mask[flat_2dim(blockIdx.x, seq_idx, seq_len)] ==
        __float2half(CUDA_FLOAT_INF_NEG)) {
      break;
    }
    for (int cur_tag = threadIdx.y; cur_tag < num_tags; cur_tag += blockDim.y) {
      float max_score = REDUCE_FLOAT_INF_NEG;
      int idx = 0;
      const __half* cur_transition = transition + cur_tag * num_tags;
      for (int pre_tag = threadIdx.x; pre_tag < num_tags;
           pre_tag += blockDim.x) {
        float s = (float)s_score[pre_tag] + (float)cur_transition[pre_tag];
        if (s > max_score) {
          max_score = s;
          idx = pre_tag;
        }
      }  // col
      g.sync();
      warp_reduce_max(g, &max_score, &idx);
      if (threadIdx.x == 0) {
        float linear_bias = bias ? float(bias[cur_tag]) : float(0);
        s_next_score[cur_tag] =
            max_score +
            float(emission[flat_3dim(blockIdx.x, seq_idx, cur_tag, seq_len,
                                     num_tags)]) +
            linear_bias;
        history[flat_3dim(blockIdx.x, seq_idx - 1, cur_tag, seq_len,
                          num_tags)] = idx;
      }
    }  // row
    float* tmp = s_next_score;
    s_next_score = s_score;
    s_score = tmp;
    b.sync();
  }  // seq_len

  // step 3. compute last tag
  if (threadIdx.y != 0) {
    return;
  }
  float max_score = REDUCE_FLOAT_INF_NEG;
  int last_tag = 0;
  for (int cur_tag = threadIdx.x; cur_tag < num_tags; cur_tag += blockDim.x) {
    float s = (float)s_score[cur_tag] + (float)end_transition[cur_tag];
    if (s > max_score) {
      max_score = s;
      last_tag = cur_tag;
    }
  }
  g.sync();
  warp_reduce_max(g, &max_score, &last_tag);

  // step 4. compute full tag sequence
  if (threadIdx.x != 0) {
    return;
  }
  if (best_score) {
    best_score[blockIdx.x] = max_score;  // for debug
  }
  seq_idx--;
  best_tags[flat_2dim(blockIdx.x, seq_idx, seq_len)] = last_tag;
  for (int i = seq_idx - 1; i >= 0; i--) {
    last_tag = history[flat_3dim(blockIdx.x, i, last_tag, seq_len, num_tags)];
    best_tags[flat_2dim(blockIdx.x, i, seq_len)] = last_tag;
  }
  for (int i = seq_idx + 1; i < seq_len; i++) {
    best_tags[flat_2dim(blockIdx.x, i, seq_len)] = invalid_tag;
  }
}

template <>
void launch_viterbi<__half>(const __half* start_transition,
                            const __half* end_transition,
                            const __half* transition, const __half* emission,
                            const __half* mask, float* best_score, int* history,
                            int* best_tags, int num_tags, int seq_len,
                            int batch_size, cudaStream_t stream,
                            const __half* bias) {
  dim3 grid_dim(batch_size);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);

  ker_viterbi<__half>
      <<<grid_dim, block_dim, 2 * num_tags * sizeof(float), stream>>>(
          start_transition, end_transition, transition, emission, mask, bias,
          best_score, history, best_tags, num_tags, seq_len);
}

template <>
void launch_viterbi<float>(const float* start_transition,
                           const float* end_transition, const float* transition,
                           const float* emission, const float* mask,
                           float* best_score, int* history, int* best_tags,
                           int num_tags, int seq_len, int batch_size,
                           cudaStream_t stream, const float* bias) {
  dim3 grid_dim(batch_size);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);

  ker_viterbi<float>
      <<<grid_dim, block_dim, 2 * num_tags * sizeof(float), stream>>>(
          start_transition, end_transition, transition, emission, mask, bias,
          best_score, history, best_tags, num_tags, seq_len);
}

}  // namespace cuda
}  // namespace lightseq
