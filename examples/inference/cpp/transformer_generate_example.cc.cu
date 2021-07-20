#include <algorithm>

#include "../../../lightseq/inference/model/decoder.h"
#include "../../../lightseq/inference/model/encoder.h"
#include "../../../lightseq/inference/tools/util.h"

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
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 0;
  }
  // when use sampling methods for decoding, set beam_size=1 to save memory
  if (tw_._sampling_method != "beam_search") {
    tw_._beam_size = 1;
  }
  tw_.print_model_config();

  /*
    step3. instantiate encoder and decoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */
  // instantiate encoder
  int max_batch_size = 8;
  thrust::device_vector<int> d_input_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<int> d_padding_mask_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<int> d_encoder_output_ =
      std::vector<int>(max_batch_size * tw_._max_step * tw_._hidden_size, 0);
  thrust::device_vector<int> d_output_ =
      std::vector<int>(max_batch_size * tw_._beam_size * tw_._max_step, 0);
  std::shared_ptr<lightseq::cuda::Encoder<optype>> encoder_ =
      std::make_shared<lightseq::cuda::Encoder<optype>>(
          max_batch_size,
          reinterpret_cast<int *>(thrust::raw_pointer_cast(d_input_.data())),
          reinterpret_cast<int *>(
              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<optraits::DataType *>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          tw_, stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 1;
  }
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
  long buf_bytesize = std::max(encoder_->compute_buffer_bytesize(),
                               decoder_->compute_buffer_bytesize());
  // thrust::device_vector<int> d_buf_ =
  //     std::vector<int>(buf_bytesize / sizeof(int), 0);
  void *d_buf_;
  // encoder and decoder use the same buffer to save gpu memory useage
  lightseq::cuda::CHECK_GPU_ERROR(cudaMalloc(&d_buf_, buf_bytesize));
  // encoder and decoder use the same buffer to save gpu memory useage
  encoder_->init_buffer(d_buf_);
  decoder_->init_buffer(d_buf_);
  cudaStreamSynchronize(stream_);

  /* ---step4. read input token ids from file--- */
  int batch_size;
  int batch_seq_len;
  std::vector<int> host_input;
  // the first line of input file should
  // be two integers: batch_size and batch_seq_len.
  // followed by batch_size lines of
  // batch_seq_len integers, e.g.
  // 2 3
  // 666 666 666
  // 666 666 666
  std::string input_file_name = argv[2];
  lightseq::cuda::read_batch_tokenids_from_file(input_file_name, batch_size,
                                                batch_seq_len, host_input);

  /* ---step5. infer and log--- */
  auto start = std::chrono::high_resolution_clock::now();
  int sum_sample_step = 0;
  for (int i = 0; i < 1; i++) {
    // copy inputs from cpu memory to gpu memory
    cudaMemcpyAsync(
        reinterpret_cast<int *>(thrust::raw_pointer_cast(d_input_.data())),
        host_input.data(), sizeof(int) * batch_size * batch_seq_len,
        cudaMemcpyHostToDevice, stream_);
    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    sum_sample_step += decoder_->_cur_step;
    for (int ii = 0; ii < batch_size; ii++) {
      for (int j = 0; j < tw_._beam_size; j++) {
        lightseq::cuda::print_vec(
            d_output_.data() + ii * tw_._beam_size * (decoder_->_cur_step + 1) +
                j * (decoder_->_cur_step + 1),
            "Beam result", decoder_->_cur_step + 1);
      }
    }
  }
  lightseq::cuda::print_time_duration(start, "Infer time", stream_);
  std::cout << "Total sampled steps: " << sum_sample_step << std::endl;
  return 0;
}
