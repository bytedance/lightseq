#include "transformer.h"

/**
@file
Example of how to run transformer inference using our implementation.
*/

int main(int argc, char *argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 8;
  //   std::string res = tw_.initializing(model_weights_path);
  //   if (!res.empty()) {
  //     std::cout << res << std::endl;
  //     return 0;
  //   }
  auto model = lightseq::cuda::Transformer(model_weights_path, max_batch_size);
  /*
    step3. instantiate encoder and decoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */
  // instantiate encoder
  //   int max_batch_size = 8;
  //   thrust::device_vector<int> d_input_ =
  //       std::vector<int>(max_batch_size * tw_._max_step, 0);
  //   thrust::device_vector<int> d_padding_mask_ =
  //       std::vector<int>(max_batch_size * tw_._max_step, 0);
  //   thrust::device_vector<int> d_encoder_output_ =
  //       std::vector<int>(max_batch_size * tw_._max_step * tw_._hidden_size,
  //       0);
  //   thrust::device_vector<int> d_output_ =
  //       std::vector<int>(max_batch_size * tw_._max_step, 0);
  //   std::shared_ptr<lightseq::cuda::Encoder<OPTYPE>> encoder_ =
  //       std::make_shared<lightseq::cuda::Encoder<OPTYPE>>(
  //           max_batch_size,
  //           reinterpret_cast<int
  //           *>(thrust::raw_pointer_cast(d_input_.data())),
  //           reinterpret_cast<int *>(
  //               thrust::raw_pointer_cast(d_padding_mask_.data())),
  //           reinterpret_cast<optraits::DataType *>(
  //               thrust::raw_pointer_cast(d_encoder_output_.data())),
  //           tw_, stream_, hd_);
  //   res = encoder_->check();
  //   if (!res.empty()) {
  //     std::cout << res << std::endl;
  //     return 1;
  //   }
  // instantiate decoder
  //   std::shared_ptr<lightseq::cuda::Decoder<OPTYPE>> decoder_ =
  //       std::make_shared<lightseq::cuda::Decoder<OPTYPE>>(
  //           max_batch_size,
  //           reinterpret_cast<int *>(
  //               thrust::raw_pointer_cast(d_padding_mask_.data())),
  //           reinterpret_cast<optraits::DataType *>(
  //               thrust::raw_pointer_cast(d_encoder_output_.data())),
  //           reinterpret_cast<int
  //           *>(thrust::raw_pointer_cast(d_output_.data())), tw_, stream_,
  //           hd_, false, reinterpret_cast<int
  //           *>(thrust::raw_pointer_cast(d_input_.data())));
  //   res = decoder_->check();
  //   if (!res.empty()) {
  //     std::cout << res << std::endl;
  //     return 1;
  //   }
  // init gpu memory buffer
  //   long buf_bytesize = std::max(encoder_->compute_buffer_bytesize(),
  //                                decoder_->compute_buffer_bytesize());
  //   thrust::device_vector<int> d_buf_ =
  //       std::vector<int>(buf_bytesize / sizeof(int), 0);
  //   // encoder and decoder use the same buffer to save gpu memory useage
  //   encoder_->init_buffer(
  //       reinterpret_cast<void *>(thrust::raw_pointer_cast(d_buf_.data())));
  //   decoder_->init_buffer(
  //       reinterpret_cast<void *>(thrust::raw_pointer_cast(d_buf_.data())));
  //   cudaStreamSynchronize(stream_);

  /* ---step4. read input token ids from file--- */
  int batch_size = 1;
  int batch_seq_len = 14;
  std::vector<int> host_input = {0,     100, 657, 14,    1816, 6, 53,
                                 50264, 473, 45,  50264, 162,  4, 2};
  // the first line of input file should
  // be two integers: batch_size and batch_seq_len.
  // followed by batch_size lines of
  // batch_seq_len integers, e.g.
  // 2 3
  // 666 666 666
  // 666 666 666
  //   std::string input_file_name = argv[2];
  //   lightseq::cuda::read_batch_tokenids_from_file(input_file_name,
  //   batch_size,
  //                                                 batch_seq_len, host_input);

  int *d_input;
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len));
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));
  /* ---step5. infer and log--- */
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // copy inputs from cpu memory to gpu memory
    // encoder_->run_one_infer(batch_size, batch_seq_len);
    // decoder_->run_one_infer(batch_size, batch_seq_len);
    model.infer(d_input, batch_size, batch_seq_len);
    lightseq::cuda::print_time_duration(start, "one infer time", 0);
    // for (int ii = 0; ii < batch_size; ii++) {
    //   lightseq::cuda::print_vec(
    //       d_output_.data() + ii * (decoder_->_cur_step + 1), "finial res",
    //       decoder_->_cur_step + 1);
    // }
  }
  return 0;
}
