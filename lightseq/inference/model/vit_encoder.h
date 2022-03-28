#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "../proto/vit_weight.h"
#include "../tools/util.h"

/**
@file
ViT encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
class VitEncoder {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  const cudaDataType_t _computeType = _optraits::computeType;
  const cudaDataType_t _AType = _optraits::AType;
  const cudaDataType_t _BType = _optraits::BType;
  const cudaDataType_t _CType = _optraits::CType;

  // private member function
  void self_attention();
  void ffn_add_norm();

  const int _max_batch_size;
  int *_p_d_padding_mask;  // true sequence length(remove padding), [batch_size]

  const VitWeight<OpType_> &_tw;
  cudaStream_t _stream;
  cublasHandle_t _hd;
  const _DataType _fone;
  const _DataType _fzero;
  const _DataType _atten_scaler;
  const int _max_batch_dim;
  const int _max_thread_per_block;

  _DataType *_p_d_qkv_projected;
  _DataType *_p_d_q;
  _DataType *_p_d_k;
  _DataType *_p_d_v;
  _DataType *_p_d_c;
  _DataType *_p_d_ffn_buf1;
  _DataType *_p_d_ffn_buf2;

  // {conv_weight, conv_bias, pos_emb, cls_embedding}
  const std::vector<const _DataType *> &_p_d_src_emb_wei;
  // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
  // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
  // ffn_norm_scale, ffn_norm_bias}
  // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
  // encoder_layer_num
  const std::vector<const _DataType *> &_p_d_enc_wei;

  int _batch_size;
  int _batch_seq_len;
  int _batch_token_num;
  int _layer_id;
  int _weight_offset;

 public:
  const float *_p_d_pixel_input;  // input pixels [batch_size, channel_input,
                                  // image_size, image_size]
  _DataType
      *_p_d_output;  // encoder output, [batch_size, batch_seq_len, hidden_size]

  VitEncoder(int max_batch_size, const float *p_d_pixel_input,
             int *p_d_padding_mask, _DataType *p_d_output,
             const VitWeight<OpType_> &tw, cudaStream_t stream,
             cublasHandle_t hd);
  long compute_buffer_bytesize();
  void init_buffer(void *pbuf);
  std::string check();
  void run_one_infer(int batch_size);
};

}  // namespace cuda
}  // namespace lightseq
