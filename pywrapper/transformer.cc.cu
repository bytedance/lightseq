#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "model/decoder.h"
#include "model/encoder.h"
#include "proto/transformer_weight.h"
#include "tools/util.h"

#ifdef FP16_MODE
const byseqlib::cuda::OperationType transformer_optytpe =
    byseqlib::cuda::OperationType::FP16;
#else
const byseqlib::cuda::OperationType transformer_optytpe =
    byseqlib::cuda::OperationType::FP32;
#endif

namespace py = pybind11;

namespace byseqlib::cuda {
class Transformer {
 private:
  typedef byseqlib::cuda::OperationTypeTraits<transformer_optytpe> optraits;
  byseqlib::cuda::Encoder<transformer_optytpe> *encoder_;
  byseqlib::cuda::Decoder<transformer_optytpe> *decoder_;

  optraits::DataType *d_encoder_output_;
  int *d_input_;
  int *d_output_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  byseqlib::cuda::TransformerWeight<transformer_optytpe> tw_;

 public:
  Transformer(const std::string weight_path, const int max_batch_size)
      : stream_(nullptr), hd_(nullptr), decoder_(nullptr) {
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

    if (tw_._sampling_method == "topk" || tw_._sampling_method == "topp") {
      tw_._beam_size = 1;
    }
    tw_.print_model_config();

    /*
      step3. instantiate encoder and decoder, init the gpu memory buffer.
        using thrust vector to avoid manage gpu memory by hand
    */

    // register device memory for inputs and outputs
    byseqlib::cuda::CHECK_GPU_ERROR(
        cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int)));
    byseqlib::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_padding_mask_, _max_batch_size * tw_._max_step * sizeof(int)));

    byseqlib::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_encoder_output_, _max_batch_size * tw_._max_step * tw_._hidden_size *
                                sizeof(optraits::DataType)));
    byseqlib::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_output_,
        _max_batch_size * tw_._beam_size * tw_._max_step * sizeof(int)));

    encoder_ = new byseqlib::cuda::Encoder<transformer_optytpe>(
        max_batch_size, d_input_, d_padding_mask_, d_encoder_output_, tw_,
        stream_, hd_);
    res = encoder_->check();
    if (!res.empty()) {
      throw std::runtime_error(res);
    }

    decoder_ = new byseqlib::cuda::Decoder<transformer_optytpe>(
        _max_batch_size, d_padding_mask_, d_encoder_output_, d_output_, tw_,
        stream_, hd_, true);
    res = decoder_->check();
    if (!res.empty()) {
      throw std::runtime_error(res);
    }

    long buf_bytesize = std::max(encoder_->compute_buffer_bytesize(),
                                 decoder_->compute_buffer_bytesize());
    std::cout << "transformer buf_bytesize: " << buf_bytesize << std::endl;

    void *d_buf_;
    // encoder and decoder use the same buffer to save gpu memory useage
    byseqlib::cuda::CHECK_GPU_ERROR(
        cudaMalloc((void **)&d_buf_, (size_t)buf_bytesize));
    encoder_->init_buffer(d_buf_);
    decoder_->init_buffer(d_buf_);
    cuerr = cudaStreamSynchronize(stream_);
    if (cuerr != cudaSuccess) {
      std::cout << "Failed to init GPU for transformer" << std::endl;
      std::runtime_error(std::string(cudaGetErrorString(cuerr)));
    }
  }

  py::array_t<int> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq) {
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int *input_seq_data = input_seq_out.data(0, 0);

    byseqlib::cuda::CHECK_GPU_ERROR(cudaMemcpyAsync(
        d_input_, input_seq_data, sizeof(int) * input_seq_out.size(),
        cudaMemcpyHostToDevice, stream_));

    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);
    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    int tokens_size = decoder_->_cur_step;
    int beam_size = tw_._beam_size;
    auto tokens = py::array_t<int>({batch_size, tokens_size});
    int *tokens_data = tokens.mutable_data(0, 0);
    byseqlib::cuda::CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_output_,
                                               sizeof(int) * tokens.size(),
                                               cudaMemcpyDeviceToHost));
    return tokens;
  }
};
}  // namespace byseqlib