#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <type_traits>

#include "cuda_util.h"

template <typename T>
class TransformerEmbeddingLayer {
 public:
  TransformerEmbeddingLayer(int layer_id, const T *pos_embeddings_ptr,
                            int max_batch_tokens, int embedding_dim,
                            int vocab_size, float dropout_ratio,
                            int padding_idx);

  virtual ~TransformerEmbeddingLayer();

  void Forward(const int *input_ptr, T *out_ptr, int step);

  void Backward(const T *grad_output_ptr, const int *input_ptr);

  void set_cur_batch_shape(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
  }

  void SetTrainingMode(bool training);
  inline bool IsTrainingMode() const { return _training; }
  inline float DropoutRatio() const { return _training ? _dropout_ratio : 0.0; }
  inline int EmbeddingDim() const { return _embedding_dim; }

  void assign_weight_ptr(const T *weights_ptr) {
    // assign weights ptr, [_vocab_size, _embedding_dim]
    _embeddings_ptr = weights_ptr;
  }

  void assign_grad_ptr(T *grads_ptr) {
    // assign grads ptr, [_vocab_size, _embedding_dim]
    _grad_embeddings_ptr = grads_ptr;
  }

 private:
  void allocate_mem_buffer() {
    // allocate local gpu memory
    _dropout_mask = cuda_malloc<uint8_t>(_max_batch_tokens * _embedding_dim);
  }

  void free_mem_buffer() {
    // free local gpu memory
    cuda_free(_dropout_mask);
  }

  // const parameter between batch
  const size_t _layer_id;
  const size_t _embedding_dim;
  const size_t _vocab_size;
  const size_t _max_batch_tokens;
  const size_t _padding_idx;
  const float _dropout_ratio;

  // dynamic parameter between batch
  size_t _batch_size;
  size_t _seq_len;
  bool _training;
  uint8_t *_dropout_mask;

  // weights ptr
  const T *_pos_embeddings_ptr;
  const T *_embeddings_ptr;

  // grads ptr
  T *_grad_embeddings_ptr;
};
