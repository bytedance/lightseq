#include "transformer_embedding_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
TransformerEmbeddingLayer<T>::TransformerEmbeddingLayer(
    int layer_id, const T *pos_embeddings_ptr, int max_batch_tokens,
    int embedding_dim, int vocab_size, float dropout_ratio, int padding_idx)
    : _layer_id(layer_id),
      _pos_embeddings_ptr(pos_embeddings_ptr),
      _max_batch_tokens(max_batch_tokens),
      _embedding_dim(embedding_dim),
      _vocab_size(vocab_size),
      _padding_idx(padding_idx),
      _dropout_ratio(dropout_ratio),
      _training(true) {
  allocate_mem_buffer();
}

template <typename T>
TransformerEmbeddingLayer<T>::~TransformerEmbeddingLayer() {
  free_mem_buffer();
}

template <typename T>
void TransformerEmbeddingLayer<T>::Forward(const int *input_ptr, T *out_ptr,
                                           int step) {
  cudaStream_t stream = Context::Instance().get_stream();
  launch_lookup_scale_pos_dropout<T>(
      out_ptr, input_ptr, _embeddings_ptr, _pos_embeddings_ptr, _dropout_mask,
      _batch_size, _seq_len, _embedding_dim, _padding_idx, DropoutRatio(), step,
      stream);
}

template <typename T>
void TransformerEmbeddingLayer<T>::Backward(const T *grad_output_ptr,
                                            const int *input_ptr) {
  cudaStream_t stream = Context::Instance().get_stream();
  launch_d_lookup_scale_pos_dropout<T>(_grad_embeddings_ptr, grad_output_ptr,
                                       input_ptr, _dropout_mask, _batch_size,
                                       _seq_len, _embedding_dim, _vocab_size,
                                       _padding_idx, DropoutRatio(), stream);
}

template <typename T>
void TransformerEmbeddingLayer<T>::SetTrainingMode(bool training) {
  // Dropout will be skipped when not in training model.
  _training = training;
}

template class TransformerEmbeddingLayer<float>;
template class TransformerEmbeddingLayer<__half>;
