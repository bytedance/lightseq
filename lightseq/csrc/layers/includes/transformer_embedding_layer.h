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
                            int vocab_size, int max_seq_len,
                            float dropout_ratio, int padding_idx,
                            bool trainable_pos);

  virtual ~TransformerEmbeddingLayer();

  void Forward(const int *input_ptr, T *out_ptr, int step);

  void Backward(const T *grad_output_ptr, const int *input_ptr);

  void set_cur_batch_shape(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
  }

  void SetTrainingMode(bool training);
  void SetQuantMode(bool enable_quant);
  inline bool IsTrainingMode() const { return _training; }
  inline float DropoutRatio() const { return _training ? _dropout_ratio : 0.0; }
  inline int EmbeddingDim() const { return _embedding_dim; }

  void assign_weight_ptr(const T *weights_ptr) {
    // assign weights ptr, [_vocab_size, _embedding_dim]
    const T *wptr = weights_ptr;

    _embeddings_ptr = wptr;
    wptr += _vocab_size * _embedding_dim;
    // if trainable postional embedding, using a combined large tensor
    if (_trainable_pos) {
      _pos_embeddings_ptr = wptr;
      wptr += _max_seq_len * _embedding_dim;
    }
    _clip_max_ptr = wptr;
  }

  void assign_grad_ptr(T *grads_ptr) {
    // assign grads ptr, [_vocab_size, _embedding_dim]
    T *gptr = grads_ptr;

    _grad_embeddings_ptr = gptr;
    gptr += _vocab_size * _embedding_dim;
    // if trainable postional embedding, set grad tensor
    if (_trainable_pos) {
      _grad_pos_embeddings_ptr = gptr;
      gptr += _max_seq_len * _embedding_dim;
    } else {
      _grad_pos_embeddings_ptr = nullptr;
    }
    _grad_clip_max_ptr = gptr;
  }

 private:
  void allocate_mem_buffer() {
    // allocate local gpu memory
    _dropout_mask = cuda_malloc<uint8_t>(_max_batch_tokens * _embedding_dim);
    cudaMalloc((void **)&_tokens_position, _max_batch_tokens * 2 * sizeof(int));
  }

  void free_mem_buffer() {
    // free local gpu memory
    cuda_free(_dropout_mask);
    cuda_free(_tokens_position);
  }

  // const parameter between batch
  const size_t _layer_id;
  const size_t _embedding_dim;
  const size_t _vocab_size;
  const size_t _max_seq_len;
  const size_t _max_batch_tokens;
  const size_t _padding_idx;
  const float _dropout_ratio;

  // dynamic parameter between batch
  size_t _batch_size;
  size_t _seq_len;
  bool _training;
  bool _trainable_pos;
  bool _enable_quant;
  uint8_t *_dropout_mask;
  int *_tokens_position;

  // weights ptr
  const T *_pos_embeddings_ptr;
  const T *_embeddings_ptr;
  const T *_clip_max_ptr;

  // grads ptr
  T *_grad_pos_embeddings_ptr;
  T *_grad_embeddings_ptr;
  T *_grad_clip_max_ptr;
};
