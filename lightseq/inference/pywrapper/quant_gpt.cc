#include "quant_gpt.h"

namespace lightseq {
namespace cuda {

QuantGpt::QuantGpt(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"result"}),
      stream_(nullptr),
      hd_(nullptr),
      encoder_(nullptr),
      _max_batch_size(max_batch_size) {
  /* ---step1. init environment--- */
  CHECK_GPU_ERROR(cudaSetDevice(0));
  CHECK_GPU_ERROR(cudaStreamCreate(&stream_));
  CHECK_GPU_ERROR(cudaStreamCreate(&cache_stream_));
  CHECK_GPU_ERROR(cublasCreate(&hd_));
  CHECK_GPU_ERROR(cublasSetStream(hd_, stream_));

  /* ---step2. load model weights into GPU memory--- */

  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  /*
    step3. instantiate gpt encoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */

  // register device memory for inputs and outputs
  CHECK_GPU_ERROR(
      cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int)));
  CHECK_GPU_ERROR(
      cudaMalloc(&d_sample_id, _max_batch_size * tw_._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(&d_ppl, _max_batch_size * sizeof(float)));

  encoder_ = std::make_shared<QuantGptEncoder<gpt_optype>>(
      max_batch_size, d_input_, d_ppl, d_sample_id, tw_, stream_, cache_stream_,
      hd_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  encoder_->init_buffer();
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
}

QuantGpt::~QuantGpt() {
  CHECK_GPU_ERROR(cudaFree(d_input_));
  CHECK_GPU_ERROR(cudaFree(d_sample_id));
  CHECK_GPU_ERROR(cudaFree(d_ppl));
  CHECK_GPU_ERROR(cudaStreamDestroy(stream_));
  CHECK_GPU_ERROR(cudaStreamDestroy(cache_stream_));
  CHECK_GPU_ERROR(cublasDestroy(hd_));
}

const int* QuantGpt::get_result_ptr() { return d_sample_id; }
const float* QuantGpt::get_score_ptr() { return d_ppl; }

void QuantGpt::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  if (tw_._sampling_method == "ppl") {
    encoder_->run_one_infer(batch_size, seq_len);
    CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
    set_output_shape(0, {batch_size});
  } else if (tw_._sampling_method == "topk" || tw_._sampling_method == "topp") {
    int sampled_seq_len = encoder_->run_one_sample(batch_size, seq_len);
    CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
    set_output_shape(0, {batch_size, sampled_seq_len});
  } else {
    throw std::runtime_error("Unsupported sampling_method");
  }
}

void QuantGpt::set_input_ptr(int index, void* input_ptr) {
  switch (index) {
    case 0:
      encoder_->_p_d_token_id = static_cast<int*>(input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void QuantGpt::set_output_ptr(int index, void* output_ptr) {
  switch (index) {
    case 0:
      if (tw_._sampling_method == "ppl") {
        encoder_->_p_d_ppl = static_cast<float*>(output_ptr);
        break;
      } else if (tw_._sampling_method == "topk" ||
                 tw_._sampling_method == "topp") {
        encoder_->_p_d_sample_id = static_cast<int*>(output_ptr);
        break;

      } else {
        throw std::runtime_error("Unsupported sampling_method");
        break;
      }

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void* QuantGpt::get_output_ptr(int index) {
  switch (index) {
    case 0:
      if (tw_._sampling_method == "ppl") {
        return static_cast<void*>(encoder_->_p_d_ppl);
        break;
      } else if (tw_._sampling_method == "topk" ||
                 tw_._sampling_method == "topp") {
        return static_cast<void*>(encoder_->_p_d_sample_id);
        break;
      } else {
        throw std::runtime_error("Unsupported sampling_method");
        break;
      }

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> QuantGpt::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

std::vector<int> QuantGpt::get_output_max_shape(int index) {
  switch (index) {
    case 0:

      if (tw_._sampling_method == "ppl") {
        return {_max_batch_size};
        break;
      } else if (tw_._sampling_method == "topk" ||
                 tw_._sampling_method == "topp") {
        return {_max_batch_size, tw_._max_step};
        break;
      } else {
        throw std::runtime_error("Unsupported sampling_method");
        break;
      }

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType QuantGpt::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType QuantGpt::get_output_dtype(int index) {
  switch (index) {
    case 0:
      if (tw_._sampling_method == "ppl") {
        return DataType::kFloat32;
        break;
      } else if (tw_._sampling_method == "topk" ||
                 tw_._sampling_method == "topp") {
        return DataType::kInt32;
        break;
      } else {
        throw std::runtime_error("Unsupported sampling_method");
        break;
      }

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
