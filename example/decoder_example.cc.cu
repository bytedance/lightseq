#include <algorithm>

#include "model/decoder.h"
#include "model/encoder.h"
#include "tools/util.h"

/**
@file
Example of how to run transformer generation inference using our implementation.
*/

// Appoint precision.
const lightseq::cuda::OperationType optype =
    lightseq::cuda::OperationType::FP32;

int main(int argc, char *argv[]) {
  /* ---step1. init environment--- */
  cudaStream_t stream_;
  cublasHandle_t hd_;
  cudaSetDevice(0);
  cudaStreamCreate(&stream_);
  cublasCreate(&hd_);
  cublasSetStream(hd_, stream_);
  typedef lightseq::cuda::OperationTypeTraits<optype> optraits;

  /* ---step2. load model weights into GPU memory--- */
  lightseq::cuda::TransformerWeight<optype> tw_;
  // saved in custom proto file
  std::string model_weights_path = argv[1];

  std::string res = tw_.initializing(model_weights_path, true);
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 0;
  }
  if (tw_._sampling_method == "topk" || tw_._sampling_method == "topp") {
    tw_._beam_size = 1;
  }

  /*
    step3. instantiate encoder and decoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */
  // instantiate encoder
  int max_batch_size = 64;
  thrust::device_vector<int> d_padding_mask_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<optraits::DataType> d_encoder_output_ =
      std::vector<optraits::DataType>(
          max_batch_size * tw_._max_step * tw_._hidden_size,
          (optraits::DataType)0.0);
  thrust::device_vector<int> d_output_ =
      std::vector<int>(max_batch_size * tw_._beam_size * tw_._max_step, 0);
  // instantiate decoder
  std::shared_ptr<lightseq::cuda::Decoder<optype>> decoder_ =
      std::make_shared<lightseq::cuda::Decoder<optype>>(
          max_batch_size,
          reinterpret_cast<int *>(
              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<optraits::DataType *>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          reinterpret_cast<int *>(thrust::raw_pointer_cast(d_output_.data())),
          tw_, stream_, hd_, true);
  res = decoder_->check();
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 1;
  }

  // init gpu memory buffer
  long buf_bytesize = decoder_->compute_buffer_bytesize();
  std::cout << "decoder buf_bytesize: " << buf_bytesize << std::endl;
  thrust::device_vector<int> d_buf_ =
      std::vector<int>(buf_bytesize / sizeof(int), 0);
  // encoder and decoder use the same buffer to save gpu memory useage

  decoder_->init_buffer(
      reinterpret_cast<void *>(thrust::raw_pointer_cast(d_buf_.data())));
  cudaStreamSynchronize(stream_);

  /* ---step4. read encoder output from file--- */
  int batch_size;
  int batch_seq_len;
  std::vector<int> host_input;
  std::string encoder_output_file = argv[2];
  std::ifstream fin(encoder_output_file);
  fin >> batch_size >> batch_seq_len;
  std::vector<float> h_encoder_output(
      batch_size * batch_seq_len * tw_._hidden_size, 0);
  for (int i = 0; i < batch_size * batch_seq_len * tw_._hidden_size; i++) {
    fin >> h_encoder_output[i];
  }

  thrust::copy(h_encoder_output.begin(), h_encoder_output.end(),
               d_encoder_output_.begin());

  /* ---step5. infer and log--- */
  auto start = std::chrono::high_resolution_clock::now();
  int sum_sample_step = 0;
  for (int i = 0; i < 100; i++) {
    decoder_->run_one_infer(batch_size, batch_seq_len);
    sum_sample_step += decoder_->_cur_step + 1;
  }
  for (int ii = 0; ii < batch_size; ii++) {
    for (int j = 0; j < tw_._beam_size; j++) {
      lightseq::cuda::print_vec(
          d_output_.data() + ii * tw_._beam_size * (decoder_->_cur_step + 1) +
              j * (decoder_->_cur_step + 1),
          "Beam result: ", decoder_->_cur_step + 1);
      lightseq::cuda::print_vec(
          decoder_->_p_d_alive_seq_score + ii * tw_._beam_size + j,
          "Beam score: ", 1);
    }
  }
  lightseq::cuda::print_time_duration(start, "infer time", stream_);
  std::cout << "Total sampled steps: " << sum_sample_step << std::endl;
  return 0;
}
