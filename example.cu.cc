#include <algorithm>

#include "src/custom/transformer/model/decoder.h"
#include "src/custom/transformer/model/encoder.h"
#include "src/custom/transformer/proto/transformer_weight.h"
#include "src/custom/transformer/util.h"

int main(int argc, char* argv[]) {
  // load model weights from proto  
  cudaStream_t stream_;
  cublasHandle_t hd_;
  cudaSetDevice(0);
  cudaStreamCreate(&stream_);
  cublasCreate(&hd_);
  cublasSetStream(hd_, stream_);

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
  std::shared_ptr<lab::nmt::Encoder> encoder_ =
      std::make_shared<lab::nmt::Encoder>(
          max_batch_size,
          reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
          reinterpret_cast<int*>(
              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<float*>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          tw_, stream_, hd_);
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
          tw_, stream_, hd_);
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
  cudaStreamSynchronize(stream_);

  int batch_size = 1;
  int batch_seq_len = 32;
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // for ru2en
    //std::vector<int> host_input = {
    //    2491,  5591,  64,    35062, 35063, 35063, 35063, 35063, 15703, 208,
    //    11,    2485,  1,     8918,  64,    35062, 2491,  5591,  64,    35062,
    //    35063, 35063, 35063, 35063, 15703, 208,   11,    2485,  1,     8918,
    //    64,    35062, 2491,  5591,  64,    35062, 35063, 35063, 35063, 35063,
    //    15703, 208,   11,    2485,  1,     8918,  64,    35062, 2491,  5591,
    //    64,    35062, 35063, 35063, 35063, 35063, 15703, 208,   11,    2485,
    //    1,     8918,  64,    35062,
    //};

    // for zh2en.1.3.77.29
    std::vector<int> host_input = {5553, 1, 2518, 1612, 3774, 104, 14559, 3698, 1572, 3030, 101, 1033, 2833, 5531, 1, 2414, 4032, 6, 111, 1503, 2169, 3774, 1529, 4063, 730, 3882, 2485, 0, 7354, 348, 2, 35611};
    // std::vector<int> host_input = {7480, 18, 1, 14673, 279, 2631, 1, 13004, 505, 893, 10065, 1, 2155, 1357, 3520, 141, 1, 3680, 557, 8, 9610, 194, 549, 0, 893, 2705, 2, 35611};
    cudaMemcpyAsync(
        reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
        host_input.data(), sizeof(int) * batch_size * batch_seq_len,
        cudaMemcpyHostToDevice, stream_);
    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    lab::nmt::print_time_duration(start, "one infer time", stream_);
  }
  return 0;
}
