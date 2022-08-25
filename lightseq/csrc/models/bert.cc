#include "bert.h"

namespace lightseq {
namespace cuda {

Bert::Bert(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"encoder_output"}),
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

  tw_.print_model_config();

  /*
    step3. instantiate encoder and decoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */

  // register device memory for inputs and outputs
  CHECK_GPU_ERROR(
      cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(&d_padding_mask_,
                             _max_batch_size * tw_._max_step * sizeof(int)));

  CHECK_GPU_ERROR(cudaMalloc(
      &d_encoder_output_, _max_batch_size * tw_._max_step * tw_._hidden_size *
                              sizeof(optraits::DataType)));

  encoder_ = std::make_shared<BertEncoder<bert_optype>>(
      max_batch_size, d_input_, d_padding_mask_, d_encoder_output_, tw_,
      stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  long buf_bytesize = encoder_->compute_buffer_bytesize();
  std::cout << "Bert buf_bytesize: " << buf_bytesize << std::endl;

  // encoder and decoder use the same buffer to save gpu memory useage
  CHECK_GPU_ERROR(cudaMalloc(&d_buf_, (size_t)buf_bytesize));
  encoder_->init_buffer(d_buf_);
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
}

Bert::~Bert() {
  CHECK_GPU_ERROR(cudaFree(d_input_));
  CHECK_GPU_ERROR(cudaFree(d_padding_mask_));
  CHECK_GPU_ERROR(cudaFree(d_encoder_output_));
  CHECK_GPU_ERROR(cudaFree(d_buf_));
  CHECK_GPU_ERROR(cublasDestroy(hd_));
  CHECK_GPU_ERROR(cudaStreamDestroy(stream_));
}

void Bert::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];
  encoder_->run_one_infer(batch_size, seq_len);
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
  set_output_shape(0, {batch_size, seq_len, tw_._hidden_size});
}

void Bert::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      encoder_->_p_d_token_id = static_cast<int *>(input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Bert::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      encoder_->_p_d_output = static_cast<optraits::DataType *>(output_ptr);
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Bert::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(encoder_->_p_d_output);

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Bert::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> Bert::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step, tw_._hidden_size};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Bert::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Bert::get_output_dtype(int index) {
  switch (index) {
    case 0:
      if (bert_optype == OperationType::FP32) {
        return DataType::kFloat32;
      } else {
        return DataType::kFloat16;
      }
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
