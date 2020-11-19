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
  std::set<std::string> available_sampling_methods = {"beam_search", "topk",
                                                      "topp", "topk_greedy"};

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

  std::tuple<py::array_t<int>, py::array_t<int>> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
      bool multiple_output = false, std::string sampling_method = "",
      int beam_size = -1, float length_penalty = -1, float topp = -1,
      float topk = -1, float diverse_lambda = -1) {
    if (available_sampling_methods.find(sampling_method) !=
        available_sampling_methods.end()) {
      tw_._sampling_method = sampling_method;
    }
    if (sampling_method == "topk" || sampling_method == "topp") {
      multiple_output = false;
    }
    if (sampling_method == "topk_greedy") {
      multiple_output = true;
    }
    decoder_->_output_topk = multiple_output;
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int *input_seq_data = input_seq_out.data(0, 0);
    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);
    if (batch_size > _max_batch_size){
      throw std::runtime_error("batch size of input greater than max_batch_size");
    }
      byseqlib::cuda::CHECK_GPU_ERROR(cudaMemcpyAsync(
          d_input_, input_seq_data, sizeof(int) * input_seq_out.size(),
          cudaMemcpyHostToDevice, stream_));

    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    int tokens_size = decoder_->_cur_step + 1;
    beam_size = tw_._beam_size;
    int output_k = multiple_output ? beam_size : 1;
    auto tokens = py::array_t<int>({batch_size, output_k, tokens_size});
    int *tokens_data = tokens.mutable_data(0, 0);
    byseqlib::cuda::CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_output_,
                                               sizeof(int) * tokens.size(),
                                               cudaMemcpyDeviceToHost));
    auto scores = py::array_t<int>({batch_size, output_k});
    // int *scores_data = scores.mutable_data(0, 0);
    // byseqlib::cuda::CHECK_GPU_ERROR(
    //     cudaMemcpy(scores_data, decoder_->_p_d_alive_seq_score,
    //                sizeof(int) * scores.size(), cudaMemcpyDeviceToHost));
    return std::make_tuple(tokens, scores);
  }
};
}  // namespace byseqlib::cuda