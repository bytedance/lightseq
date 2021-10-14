// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>

// #include "../model/decoder.h"
// #include "../model/encoder.h"
// #include "../proto/transformer_weight.h"
// #include "../tools/util.h"
#include "transformer.h"

// #ifdef FP16_MODE
// const lightseq::cuda::OperationType transformer_optytpe =
//     lightseq::cuda::OperationType::FP16;
// #else
// const lightseq::cuda::OperationType transformer_optytpe =
//     lightseq::cuda::OperationType::FP32;
// #endif

namespace lightseq {
namespace cuda {

Transformer::Transformer(const std::string weight_path,
                         const int max_batch_size)
    : stream_(nullptr),
      hd_(nullptr),
      decoder_(nullptr),
      _max_batch_size(max_batch_size) {
  /* ---step1. init environment--- */
  CHECK_GPU_ERROR(cudaSetDevice(0));
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

  // register device memory for inputs and outputs
  CHECK_GPU_ERROR(
      cudaMalloc(&d_input_, _max_batch_size * tw_._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(&d_padding_mask_,
                             _max_batch_size * tw_._max_step * sizeof(int)));

  CHECK_GPU_ERROR(cudaMalloc(
      &d_encoder_output_, _max_batch_size * tw_._max_step * tw_._hidden_size *
                              sizeof(optraits::DataType)));
  CHECK_GPU_ERROR(cudaMalloc(&d_output_, _max_batch_size * tw_._beam_size *
                                             tw_._max_step * sizeof(int)));

  encoder_ = std::make_shared<Encoder<transformer_optytpe>>(
      _max_batch_size, d_input_, d_padding_mask_, d_encoder_output_, tw_,
      stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  decoder_ = std::make_shared<Decoder<transformer_optytpe>>(
      _max_batch_size, d_padding_mask_, d_encoder_output_, d_output_, tw_,
      stream_, hd_, true);
  res = decoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  long buf_bytesize = std::max(encoder_->compute_buffer_bytesize(),
                               decoder_->compute_buffer_bytesize());
  std::cout << "Allocated " << buf_bytesize / (1024 * 1024)
            << "MB GPU buffer for transformer" << std::endl;

  // encoder and decoder use the same buffer to save gpu memory useage
  CHECK_GPU_ERROR(cudaMalloc(&d_buf_, buf_bytesize));
  encoder_->init_buffer(d_buf_);
  decoder_->init_buffer(d_buf_);
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
}

Transformer::~Transformer() {
  CHECK_GPU_ERROR(cudaFree(d_input_));
  CHECK_GPU_ERROR(cudaFree(d_padding_mask_));
  CHECK_GPU_ERROR(cudaFree(d_encoder_output_));
  CHECK_GPU_ERROR(cudaFree(d_output_));
  CHECK_GPU_ERROR(cudaFree(d_buf_));
  CHECK_GPU_ERROR(cudaStreamDestroy(stream_));
}

const int *Transformer::get_result_ptr() { return d_output_; }

const float *Transformer::get_score_ptr() {
  return decoder_->_p_d_alive_seq_score;
}

int Transformer::get_output_seq_len() { return decoder_->_cur_step + 1; };

#ifdef ENABLE_PYTHON

std::tuple<py::array_t<int>, py::array_t<float>> Transformer::infer(
    py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
    bool multiple_output) {
  auto input_seq_out = input_seq.mutable_unchecked<2>();
  const int *input_seq_data = input_seq_out.data(0, 0);
  int batch_size = input_seq_out.shape(0);
  int batch_seq_len = input_seq_out.shape(1);

  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpyAsync(
      d_input_, input_seq_data, sizeof(int) * input_seq_out.size(),
      cudaMemcpyHostToDevice, stream_));

  encoder_->run_one_infer(batch_size, batch_seq_len);
  decoder_->run_one_infer(batch_size, batch_seq_len);
  int tokens_size = get_output_seq_len();
  int beam_size = tw_._beam_size;
  int output_k = multiple_output ? beam_size : 1;
  auto tokens = py::array_t<int>({batch_size, output_k, tokens_size});
  int *tokens_data = tokens.mutable_data(0, 0);
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_output_,
                                             sizeof(int) * tokens.size(),
                                             cudaMemcpyDeviceToHost));
  auto scores = py::array_t<float>({batch_size, output_k});
  float *scores_data = scores.mutable_data(0, 0);
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMemcpy(scores_data, decoder_->_p_d_alive_seq_score,
                 sizeof(float) * scores.size(), cudaMemcpyDeviceToHost));
  return std::make_tuple(tokens, scores);
}
#else

std::tuple<int, int, int> Transformer::infer(int *input_seq, int batch_size,
                                             int batch_seq_len, int *result_seq,
                                             float *scores,
                                             bool multiple_output) {
  if (multiple_output) {
    if (tw_._sampling_method == "beam_search") {
      decoder_->_output_topk = multiple_output;
    } else {
      std::cout << "multiple_output will only work on beam search" << std::endl;
    }
  }

  int *old_input_ptr = encoder_->_p_d_token_id;
  encoder_->_p_d_token_id = input_seq;

  int *old_result_ptr = nullptr;
  if (result_seq != nullptr) {
    old_result_ptr = decoder_->_p_d_result;
    decoder_->_p_d_result = result_seq;
  }

  float *old_score_ptr = nullptr;
  if (scores != nullptr) {
    old_score_ptr = decoder_->_p_d_alive_seq_score;
    decoder_->_p_d_alive_seq_score = scores;
  }

  encoder_->run_one_infer(batch_size, batch_seq_len);
  decoder_->run_one_infer(batch_size, batch_seq_len);

  CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));

  int output_seq_len = get_output_seq_len();
  int beam_size = tw_._beam_size;
  int output_k = decoder_->_output_topk ? beam_size : 1;

  if (old_result_ptr != nullptr) {
    decoder_->_p_d_result = old_result_ptr;
  }
  if (old_score_ptr != nullptr) {
    decoder_->_p_d_alive_seq_score = old_score_ptr;
  }
  encoder_->_p_d_token_id = old_input_ptr;

  return std::make_tuple(batch_size, output_k, output_seq_len);
}
#endif
}  // namespace cuda
}  // namespace lightseq
