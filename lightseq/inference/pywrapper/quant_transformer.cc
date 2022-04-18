#include "quant_transformer.h"

#include "embKernels_int8.h"

namespace lightseq {
namespace cuda {

QuantTransformer::QuantTransformer(const std::string weight_path,
                                   const int max_batch_size)
    : LSModel({"source_ids"}, {"target_ids", "target_scores"}),
      stream_(nullptr),
      hd_(nullptr),
      decoder_(nullptr),
      _max_batch_size(max_batch_size) {
  /* ---step1. init environment--- */
  CHECK_GPU_ERROR(cudaStreamCreate(&stream_));
  CHECK_GPU_ERROR(cublasCreate(&hd_));
  CHECK_GPU_ERROR(cublasSetStream(hd_, stream_));

  /* ---step2. load model weights into GPU memory--- */

  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  if (tw_._sampling_method == "topk" || tw_._sampling_method == "topp") {
    tw_._beam_size = 1;
  }
  tw_.print_model_config();

  /*
    step3. instantiate encoder and decoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */

  CHECK_GPU_ERROR(
      cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int32_t)));
  CHECK_GPU_ERROR(cudaMalloc(
      &d_padding_mask_, _max_batch_size * tw_._max_step * sizeof(int32_t)));

  CHECK_GPU_ERROR(cudaMalloc(
      &d_encoder_output_, _max_batch_size * tw_._max_step * tw_._hidden_size *
                              sizeof(optraits::DataType)));
  CHECK_GPU_ERROR(
      cudaMalloc(&d_src_lang_id_, _max_batch_size * sizeof(int32_t)));
  CHECK_GPU_ERROR(
      cudaMalloc(&d_trg_lang_id_, _max_batch_size * sizeof(int32_t)));

  encoder_ = std::make_shared<QuantEncoder<qtransformer_optytpe>>(
      _max_batch_size, d_input_, d_padding_mask_, d_encoder_output_, tw_,
      stream_, hd_, d_src_lang_id_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  decoder_ = std::make_shared<QuantDecoder<qtransformer_optytpe>>(
      _max_batch_size, d_padding_mask_, d_encoder_output_, d_output_, tw_,
      stream_, hd_, true, d_trg_lang_id_);
  res = decoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  encoder_->init_buffer();
  decoder_->init_buffer();
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
}

QuantTransformer::~QuantTransformer() {
  CHECK_GPU_ERROR(cudaFree(d_input_));
  CHECK_GPU_ERROR(cudaFree(d_padding_mask_));
  CHECK_GPU_ERROR(cudaFree(d_encoder_output_));
  CHECK_GPU_ERROR(cudaFree(d_src_lang_id_));
  CHECK_GPU_ERROR(cudaFree(d_trg_lang_id_));
  CHECK_GPU_ERROR(cudaStreamDestroy(stream_));
}

const int *QuantTransformer::get_result_ptr() { return d_output_; }

const float *QuantTransformer::get_score_ptr() {
  return decoder_->_p_d_alive_seq_score;
}

int QuantTransformer::get_output_seq_len() { return decoder_->_cur_step + 1; };

void QuantTransformer::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  // for multilg
  if (tw_._multilg_type != 0) {
    throw std::runtime_error("multilingle not supported");
  }

  encoder_->run_one_infer(batch_size, seq_len);
  decoder_->run_one_infer(batch_size, seq_len);

  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));

  int output_seq_len = get_output_seq_len();
  int beam_size = tw_._beam_size;
  int output_k = decoder_->_output_topk ? beam_size : 1;

  set_output_shape(0, {batch_size, output_k, output_seq_len});
  set_output_shape(1, {batch_size, output_k});
}

void QuantTransformer::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      encoder_->_p_d_token_id = static_cast<int *>(input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void QuantTransformer::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      decoder_->_p_d_result = static_cast<int *>(output_ptr);
      break;

    case 1:
      decoder_->_p_d_alive_seq_score = static_cast<float *>(output_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
const void *QuantTransformer::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(decoder_->_p_d_result);
      break;

    case 1:
      return static_cast<void *>(decoder_->_p_d_alive_seq_score);
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> QuantTransformer::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

std::vector<int> QuantTransformer::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._beam_size, tw_._max_step};
      break;

    case 1:
      return {_max_batch_size, tw_._beam_size};
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType QuantTransformer::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType QuantTransformer::get_output_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    case 1:
      return DataType::kFloat32;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
