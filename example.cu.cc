#include <algorithm>

#include "src/custom/transformer/model/decoder.h"
#include "src/custom/transformer/model/encoder.h"
#include "src/custom/transformer/proto/transformer_weight.h"
#include "src/custom/transformer/util.h"

int main(int argc, char* argv[]) {
  // load model weights from proto
  lab::nmt::TransformerWeight tw_;
  std::string res = tw_.initializing(argv[1]);  // proto path
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 0;
  }
  // tw_._length_penalty = 0.6;

  // init encoder and decoder
  // use thrust vector to avoid manage gpu memory by hand
  int max_batch_size = 8;
  int datatype_bytesize_ = 4;
  thrust::device_vector<int> d_input_ =
      std::vector<int>(max_batch_size * tw_._max_step * datatype_bytesize_, 0);
  thrust::device_vector<int> d_padding_mask_ =
      std::vector<int>(max_batch_size * tw_._max_step * datatype_bytesize_, 0);
  thrust::device_vector<int> d_encoder_output_ = std::vector<int>(
      max_batch_size * tw_._max_step * tw_._hidden_size * datatype_bytesize_,
      0);
  thrust::device_vector<int> d_output_ =
      std::vector<int>(max_batch_size * tw_._max_step * datatype_bytesize_, 0);
  cublasHandle_t hd_;
  CUBLAS_CALL(cublasCreate(&hd_));
  std::shared_ptr<lab::nmt::Encoder> encoder_ =
      std::make_shared<lab::nmt::Encoder>(
          max_batch_size,
          reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
          reinterpret_cast<int*>(
              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<float*>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          tw_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 1;
  }
  std::shared_ptr<lab::nmt::Decoder> decoder_ =
      std::make_shared<lab::nmt::Decoder>(
          max_batch_size, reinterpret_cast<int*>(
                              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<float*>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          reinterpret_cast<int*>(thrust::raw_pointer_cast(d_output_.data())),
          tw_, hd_);
  res = decoder_->check();
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 1;
  }
  int buf_bytesize = std::max(encoder_->compute_buffer_bytesize(),
                              decoder_->compute_buffer_bytesize());
  thrust::device_vector<int> d_buf_ =
      std::vector<int>(buf_bytesize / sizeof(int), 0);
  // encoder and decoder use the same buffer to save gpu memory useage
  encoder_->init_buffer(
      reinterpret_cast<void*>(thrust::raw_pointer_cast(d_buf_.data())));
  decoder_->init_buffer(
      reinterpret_cast<void*>(thrust::raw_pointer_cast(d_buf_.data())));
  cudaDeviceSynchronize();

  int batch_size = 8;
  int batch_seq_len = 8;
  for (int i = 0; i < 5; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> host_input = {
        2491,  5591,  64,    35062, 35063, 35063, 35063, 35063, 15703, 208,
        11,    2485,  1,     8918,  64,    35062, 2491,  5591,  64,    35062,
        35063, 35063, 35063, 35063, 15703, 208,   11,    2485,  1,     8918,
        64,    35062, 2491,  5591,  64,    35062, 35063, 35063, 35063, 35063,
        15703, 208,   11,    2485,  1,     8918,  64,    35062, 2491,  5591,
        64,    35062, 35063, 35063, 35063, 35063, 15703, 208,   11,    2485,
        1,     8918,  64,    35062,
    };
    cudaMemcpy(
        reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
        host_input.data(), sizeof(int) * batch_size * batch_seq_len,
        cudaMemcpyHostToDevice);
    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    lab::nmt::print_time_duration(start, "one infer time");
  }
  return 0;
}
