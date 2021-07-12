#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../model/gpt_encoder.h"
#include "../proto/gpt_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace py = pybind11;

namespace lightseq {
namespace cuda {
class Gpt {
 private:
  typedef lightseq::cuda::OperationTypeTraits<gpt_optype> optraits;
  lightseq::cuda::GptEncoder<gpt_optype>* encoder_;

  int* d_input_;
  int* d_sample_id;
  float* d_ppl;
  int _max_batch_size;
  cudaStream_t stream_;
  cudaStream_t cache_stream_;
  cublasHandle_t hd_;
  lightseq::cuda::GptWeight<gpt_optype> tw_;
  std::set<std::string> available_sampling_methods = {"topk", "topp"};

 public:
  Gpt(const std::string weight_path, const int max_batch_size)
      : stream_(nullptr), hd_(nullptr), encoder_(nullptr) {
    /* ---step1. init environment--- */
    _max_batch_size = max_batch_size;
    cudaError_t cuerr = cudaSetDevice(0);
    if (cuerr != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cuerr));
    }
    cuerr = cudaStreamCreate(&stream_);
    if (cuerr != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cuerr));
    }
    cuerr = cudaStreamCreate(&cache_stream_);
    if (cuerr != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cuerr));
    }
    cublasStatus_t cublaserr = cublasCreate(&hd_);
    if (cublaserr != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to creat cublas handle ");
    }
    cublaserr = cublasSetStream(hd_, stream_);
    if (cublaserr != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to set stream for cublas handle");
    }

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
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int)));
    lightseq::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_sample_id, _max_batch_size * tw_._max_step * sizeof(int)));
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMalloc(&d_ppl, _max_batch_size * sizeof(float)));

    encoder_ = new lightseq::cuda::GptEncoder<gpt_optype>(
        max_batch_size, d_input_, d_ppl, d_sample_id, tw_, stream_,
        cache_stream_, hd_);
    res = encoder_->check();
    if (!res.empty()) {
      throw std::runtime_error(res);
    }

    size_t buf_bytesize = encoder_->compute_buffer_bytesize();
    std::cout << "Allocated " << buf_bytesize / (1024 * 1024)
              << "MB GPU buffer for GPT2" << std::endl;

    void* d_buf_;
    // encoder and decoder use the same buffer to save gpu memory useage
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMalloc((void**)&d_buf_, (size_t)buf_bytesize));
    encoder_->init_buffer(d_buf_);
    cuerr = cudaStreamSynchronize(stream_);
    if (cuerr != cudaSuccess) {
      std::cout << "Failed to init GPU for transformer" << std::endl;
      std::runtime_error(std::string(cudaGetErrorString(cuerr)));
    }
  }
  py::array_t<float> ppl(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq) {
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int* input_seq_data = input_seq_out.data(0, 0);
    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);
    if (batch_size > _max_batch_size) {
      throw std::runtime_error(
          "batch size of input greater than max_batch_size");
    }
    if (batch_seq_len > tw_._max_step) {
      throw std::runtime_error("seq len of input greater than max_step");
    }
    lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpyAsync(
        d_input_, input_seq_data, sizeof(int) * input_seq_out.size(),
        cudaMemcpyHostToDevice, stream_));

    encoder_->run_one_infer(batch_size, batch_seq_len);

    auto probs = py::array_t<float>(batch_size);
    float* probs_data = probs.mutable_data(0);
    lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(probs_data, d_ppl,
                                               sizeof(float) * probs.size(),
                                               cudaMemcpyDeviceToHost));
    return probs;
  }

  py::array_t<int> sample(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
      std::string sampling_method = "topk", const int topk = 1,
      const float topp = 0.75) {
    if (available_sampling_methods.find(sampling_method) !=
        available_sampling_methods.end()) {
      tw_._sampling_method = sampling_method;
    }
    assert(topk >= 0);
    tw_._topk = topk;
    assert(topp >= 0.0 && topp <= 1.0);
    tw_._topp = topp;

    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int* input_seq_data = input_seq_out.data(0, 0);
    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);
    if (batch_size > _max_batch_size) {
      throw std::runtime_error(
          "batch size of input greater than max_batch_size");
    }
    if (batch_seq_len > tw_._max_step) {
      throw std::runtime_error("seq len of input greater than max_step");
    }
    lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpyAsync(
        d_input_, input_seq_data, sizeof(int) * input_seq_out.size(),
        cudaMemcpyHostToDevice, stream_));

    int sampled_seq_len = encoder_->run_one_sample(batch_size, batch_seq_len);

    auto tokens = py::array_t<int>({batch_size, sampled_seq_len});
    int* tokens_data = tokens.mutable_data(0);
    lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_sample_id,
                                               sizeof(int) * tokens.size(),
                                               cudaMemcpyDeviceToHost));
    return tokens;
  }
};
}  // namespace cuda
}  // namespace lightseq
