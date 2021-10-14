#include "gpt.h"

namespace lightseq {
namespace cuda {

Gpt::Gpt(const std::string weight_path, const int max_batch_size)
    : stream_(nullptr),
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

  encoder_ =
      new GptEncoder<gpt_optype>(max_batch_size, d_input_, d_ppl, d_sample_id,
                                 tw_, stream_, cache_stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  size_t buf_bytesize = encoder_->compute_buffer_bytesize();
  std::cout << "Allocated " << buf_bytesize / (1024 * 1024)
            << "MB GPU buffer for GPT2" << std::endl;

  // encoder and decoder use the same buffer to save gpu memory useage
  CHECK_GPU_ERROR(cudaMalloc((void**)&d_buf_, (size_t)buf_bytesize));
  encoder_->init_buffer(d_buf_);
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
}

Gpt::~Gpt() {
  CHECK_GPU_ERROR(cudaFree(d_input_));
  CHECK_GPU_ERROR(cudaFree(d_sample_id));
  CHECK_GPU_ERROR(cudaFree(d_ppl));
  CHECK_GPU_ERROR(cudaFree(d_buf_));
  CHECK_GPU_ERROR(cudaStreamDestroy(stream_));
  CHECK_GPU_ERROR(cudaStreamDestroy(cache_stream_));
  CHECK_GPU_ERROR(cublasDestroy(hd_));
}

const int* Gpt::get_result_ptr() { return d_sample_id; }
const float* Gpt::get_score_ptr() { return d_ppl; }

#ifdef ENABLE_PYTHON

py::array_t<float> Gpt::ppl(
    py::array_t<int, py::array::c_style | py::array::forcecast> input_seq) {
  auto input_seq_out = input_seq.mutable_unchecked<2>();
  const int* input_seq_data = input_seq_out.data(0, 0);
  int batch_size = input_seq_out.shape(0);
  int batch_seq_len = input_seq_out.shape(1);

  CHECK_GPU_ERROR(cudaMemcpyAsync(d_input_, input_seq_data,
                                  sizeof(int) * input_seq_out.size(),
                                  cudaMemcpyHostToDevice, stream_));

  encoder_->run_one_infer(batch_size, batch_seq_len);

  auto probs = py::array_t<float>(batch_size);
  float* probs_data = probs.mutable_data(0);
  CHECK_GPU_ERROR(cudaMemcpy(probs_data, d_ppl, sizeof(float) * probs.size(),
                             cudaMemcpyDeviceToHost));
  return probs;
}

py::array_t<int> Gpt::sample(
    py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
    std::string sampling_method, const int topk, const float topp) {
  if (!sampling_method.empty()) {
    tw_._sampling_method = sampling_method;
  }
  if (topk != -1) {
    tw_._topk = topk;
  }
  if (topp != -1) {
    tw_._topp = topp;
  }

  std::string res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  auto input_seq_out = input_seq.mutable_unchecked<2>();
  const int* input_seq_data = input_seq_out.data(0, 0);
  int batch_size = input_seq_out.shape(0);
  int batch_seq_len = input_seq_out.shape(1);

  CHECK_GPU_ERROR(cudaMemcpyAsync(d_input_, input_seq_data,
                                  sizeof(int) * input_seq_out.size(),
                                  cudaMemcpyHostToDevice, stream_));

  int sampled_seq_len = encoder_->run_one_sample(batch_size, batch_seq_len);

  auto tokens = py::array_t<int>({batch_size, sampled_seq_len});
  int* tokens_data = tokens.mutable_data(0);
  CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_sample_id,
                             sizeof(int) * tokens.size(),
                             cudaMemcpyDeviceToHost));
  return tokens;
}

#else

int Gpt::ppl(const int* input_seq, int batch_size, int batch_seq_len,
             float* result_seq) {
  const int* old_input_ptr = encoder_->_p_d_token_id;
  encoder_->_p_d_token_id = input_seq;

  float* old_result_ptr = nullptr;
  if (result_seq != nullptr) {
    old_result_ptr = encoder_->_p_d_ppl;
    encoder_->_p_d_ppl = result_seq;
  }

  encoder_->run_one_infer(batch_size, batch_seq_len);

  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));

  if (result_seq != nullptr) {
    encoder_->_p_d_ppl = old_result_ptr;
  }
  return batch_size;
}

std::tuple<int, int> Gpt::sample(const int* input_seq, int batch_size,
                                 int batch_seq_len, int* result_seq) {
  const int* old_input_ptr = encoder_->_p_d_token_id;
  encoder_->_p_d_token_id = input_seq;

  int* old_result_ptr = nullptr;
  if (result_seq != nullptr) {
    old_result_ptr = encoder_->_p_d_sample_id;
    encoder_->_p_d_sample_id = result_seq;
  }

  int sampled_seq_len = encoder_->run_one_sample(batch_size, batch_seq_len);

  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));

  if (result_seq != nullptr) {
    encoder_->_p_d_sample_id = old_result_ptr;
  }
  return std::make_tuple(batch_size, sampled_seq_len);
}

#endif
}  // namespace cuda
}  // namespace lightseq
