#include <algorithm>

#include "model/gpt_encoder.h"
#include "tools/util.h"

/**
@file
Example of how to run gpt inference using our implementation.
*/

// Appoint precision.
const lightseq::cuda::OperationType optype =
    lightseq::cuda::OperationType::FP32;

int main(int argc, char *argv[]) {
  /* ---step1. init environment--- */
  cudaStream_t stream_;
  cudaStream_t cache_stream_;
  cublasHandle_t hd_;
  cudaSetDevice(0);
  cudaStreamCreate(&stream_);
  cudaStreamCreate(&cache_stream_);
  cublasCreate(&hd_);
  cublasSetStream(hd_, stream_);

  /* ---step2. load model weights into GPU memory--- */
  lightseq::cuda::GptWeight<optype> tw_;
  // saved in custom proto file
  std::string model_weights_path = argv[1];
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 0;
  }

  /*
    step3. instantiate encoder, init the gpu memory buffer.
      using thrust vector to avoid manage gpu memory by hand
  */
  int max_batch_size = 128;
  thrust::device_vector<int> d_input_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<int> d_sample_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<float> d_ppl_ = std::vector<float>(max_batch_size, 0.f);
  std::shared_ptr<lightseq::cuda::GptEncoder<optype>> encoder_ =
      std::make_shared<lightseq::cuda::GptEncoder<optype>>(
          max_batch_size,
          reinterpret_cast<int *>(thrust::raw_pointer_cast(d_input_.data())),
          reinterpret_cast<float *>(thrust::raw_pointer_cast(d_ppl_.data())),
          reinterpret_cast<int *>(thrust::raw_pointer_cast(d_sample_.data())),
          tw_, stream_, cache_stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 1;
  }
  // init gpu memory buffer
  long buf_bytesize = encoder_->compute_buffer_bytesize();
  thrust::device_vector<int> d_buf_ =
      std::vector<int>(buf_bytesize / sizeof(int) + 1, 0);
  encoder_->init_buffer(
      reinterpret_cast<void *>(thrust::raw_pointer_cast(d_buf_.data())));
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
  int sample_step;
  int sum_sample_step = 0;
  for (int i = 0; i < 100; i++) {
    // copy inputs from cpu memory to gpu memory
    cudaMemcpyAsync(
        reinterpret_cast<int *>(thrust::raw_pointer_cast(d_input_.data())),
        host_input.data(), sizeof(int) * batch_size * batch_seq_len,
        cudaMemcpyHostToDevice, stream_);
    sample_step = encoder_->run_one_sample(batch_size, batch_seq_len);
    int *sample_output = thrust::raw_pointer_cast(d_sample_.data());
    sample_output += batch_seq_len;
    sum_sample_step += sample_step - batch_seq_len;
  }
  lightseq::cuda::print_vec(d_sample_.data(), "sample_output",
                            batch_size * sample_step);
  lightseq::cuda::print_time_duration(start, "one infer time", stream_);
  std::cout << "Total sampled steps: " << sum_sample_step << std::endl;
  return 0;
}
