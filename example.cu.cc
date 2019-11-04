#include <algorithm>

#include "src/custom/transformer/model/decoder.h"
#include "src/custom/transformer/model/encoder.h"
#include "src/custom/transformer/proto/transformer_weight.h"
#include "src/custom/transformer/util.h"
const lab::nmt::OperationType optype = lab::nmt::OperationType::FP32;

int main(int argc, char* argv[]) {
  // load model weights from proto  
  cudaStream_t stream_;
  cublasHandle_t hd_;
  cudaSetDevice(0);
  cudaStreamCreate(&stream_);
  cublasCreate(&hd_);
  cublasSetStream(hd_, stream_);

  typedef lab::nmt::OperationTypeTraits<optype> optraits;

  lab::nmt::TransformerWeight<optype> tw_;
  std::string res = tw_.initializing(argv[1]);  // proto path
  if (!res.empty()) {
    std::cout << res << std::endl;
    return 0;
  }
  // tw_._length_penalty = 0.6;

  // init encoder and decoder
  // use thrust vector to avoid manage gpu memory by hand
  int max_batch_size = 8;
  thrust::device_vector<int> d_input_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<int> d_padding_mask_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  thrust::device_vector<int> d_encoder_output_ = std::vector<int>(
      max_batch_size * tw_._max_step * tw_._hidden_size,
      0);
  thrust::device_vector<int> d_output_ =
      std::vector<int>(max_batch_size * tw_._max_step, 0);
  std::shared_ptr<lab::nmt::Encoder<optype>> encoder_ =
      std::make_shared<lab::nmt::Encoder<optype>>(
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
  std::shared_ptr<lab::nmt::Decoder<optype>> decoder_ =
      std::make_shared<lab::nmt::Decoder<optype>>(
          max_batch_size, reinterpret_cast<int *>(
                              thrust::raw_pointer_cast(d_padding_mask_.data())),
          reinterpret_cast<optraits::DataType *>(
              thrust::raw_pointer_cast(d_encoder_output_.data())),
          reinterpret_cast<int *>(thrust::raw_pointer_cast(d_output_.data())),
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

  int batch_size = 8;
  int batch_seq_len = 78;
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
    // std::vector<int> host_input =
    //     {5553, 1, 2518, 1612, 3774, 104, 14559, 3698, 1572, 3030, 101, 1033,
    //     2833, 5531, 1, 2414,
    //         4032, 6, 111, 1503, 2169, 3774, 1529, 4063, 730, 3882, 2485, 0,
    //         7354, 348, 2, 35611};
    std::vector<int> host_input = {
	    1074, 509, 1, 5, 1251, 5, 58, 2741, 5579, 4, 9589, 101, 4, 1048, 0, 3311, 1745, 2,
	    10811, 40, 525, 4, 7281, 1140, 800, 1, 7, 58, 711, 18, 721, 800, 101, 355, 30, 0, 879,
	    2, 2496, 25, 6253, 7231, 149, 2369, 6, 995, 1, 43, 736, 71, 1, 4, 1829, 5, 1288, 71,
	    4017, 1, 146, 7, 513, 1131, 4806, 1, 224, 395, 4, 1328, 3052, 515, 6, 8197, 9, 2, 4164,
	    5825, 3, 28080, 2528, 1, 576, 2, 6427, 1, 15, 1107, 365, 16259, 1045, 1, 69, 19, 272, 5,
	    1099, 30, 7, 469, 14402, 13428, 5201, 1, 6, 0, 2002, 2, 0, 2591, 2832, 1, 73, 0, 81, 82,
	    1119, 5, 1268, 24, 97, 7, 11617, 3900, 162, 1, 15, 1107, 5507, 541, 1, 6597, 40, 0, 935,
	    2, 3461, 1226, 1, 38, 64, 6801, 323, 0, 16331, 1451, 3, 28080, 28081, 28081, 28081,
	    28081, 28081, 28081, 28081, 28081, 28081, 28081, 477, 1, 271, 1367, 13493, 414, 7113, 9,
	    13, 7, 196, 6415, 215, 1, 24, 6134, 71, 12, 24, 57, 43, 7, 204, 2, 0, 223, 750, 62, 0,
	    16969, 234, 4516, 913, 1, 38, 1209, 12, 153, 12385, 648, 10900, 55, 0, 223, 856, 34, 0,
	    223, 91, 96, 63, 26, 6, 0, 223, 58, 2860, 13284, 86, 559, 12182, 29, 58, 559, 101, 4875,
	    10151, 125, 2, 469, 2, 48, 2130, 3, 28080, 28081, 28081, 28081, 9292, 1374, 5334, 14952,
	    2481, 1, 1236, 213, 1001, 5656, 8, 1, 1569, 24, 6, 3270, 6, 1478, 2, 0, 3680, 62, 6,
	    1470, 166, 1, 68, 645, 30, 1267, 681, 9983, 1, 17357, 603, 5, 10742, 1, 11821, 11171, 9,
	    1, 6195, 1, 2450, 10651, 55, 8030, 9, 4, 5560, 15942, 10317, 40, 6532, 170, 77, 368,
	    2609, 1, 6, 1021, 10, 7371, 200, 1858, 2, 38, 68, 17166, 364, 0, 591, 3, 28080, 28081,
	    28081, 28081, 485, 15416, 510, 26, 273, 4180, 97, 153, 468, 1, 694, 5, 0, 5248, 9633,
	    2607, 10380, 9, 17871, 9096, 14805, 13284, 110, 915, 25, 4950, 5484, 1, 21, 0, 181,
	    1800, 7950, 510, 1, 38, 49, 41, 32, 2372, 189, 1, 4, 58, 613, 12385, 101, 21, 38, 0,
	    470, 1091, 2, 825, 26, 15464, 364, 4, 5, 38, 0, 6059, 6256, 9, 15416, 14, 109, 19, 6047,
	    210, 351, 3, 28080, 28081, 28081, 28081, 28081, 28081, 9997, 243, 15, 7, 2409, 18, 180,
	    18, 588, 331, 83, 13, 7334, 3714, 40, 79, 344, 304, 15, 79, 407, 1, 4101, 7680, 3649,
	    14074, 46, 6, 79, 6666, 12122, 346, 14756, 1966, 26, 11178, 6, 79, 3216, 1, 7, 1937, 18,
	    180, 18, 702, 331, 13, 7222, 18823, 3079, 1, 8178, 21, 79, 1227, 1, 79, 1628, 4, 79,
	    153, 702, 183, 1, 218, 487, 2225, 979, 5, 7579, 3, 28080, 28081, 28081, 28081, 28081,
	    28081, 16, 104, 299, 1866, 646, 10, 0, 6609, 2, 1407, 525, 62, 1287, 199, 292, 567, 1,
	    2775, 199, 292, 567, 1, 1131, 946, 34, 0, 1473, 1, 1131, 1641, 1, 299, 6, 0, 721, 1,
	    336, 1, 1004, 4, 1328, 1827, 1, 163, 5, 0, 12016, 11432, 1570, 8226, 8, 3, 28080, 28081,
	    28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081,
	    28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 16,
	    2391, 545, 410, 5, 19, 135, 5, 1010, 0, 1977, 2, 532, 2, 79, 2277, 1, 21, 2565, 174, 0,
	    1881, 333, 1, 32, 4534, 10962, 12249, 1361, 17339, 9, 12, 26, 32, 315, 18, 502, 238,
	    9302, 1, 4, 5303, 40, 2065, 1, 6, 94, 409, 1, 10, 982, 777, 2, 0, 1562, 3, 28080, 28081,
	    28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081,
	    28081, 28081, 28081, 28081, 28081, 28081, 28081, 28081
    };
    // std::vector<int> host_input = {7480, 18, 1, 14673, 279, 2631, 1, 13004, 505, 893, 10065, 1, 2155, 1357, 3520, 141, 1, 3680, 557, 8, 9610, 194, 549, 0, 893, 2705, 2, 35611};
    cudaMemcpyAsync(
        reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
        host_input.data(), sizeof(int) * batch_size * batch_seq_len,
        cudaMemcpyHostToDevice, stream_);
    encoder_->run_one_infer(batch_size, batch_seq_len);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    lab::nmt::print_time_duration(start, "one infer time", stream_);
    for(int ii=0; ii<batch_size; ii++) {
        lab::nmt::print_vec(d_output_.data() + ii * (decoder_->_cur_step + 1), "finial res",
        	      decoder_->_cur_step + 1);
    }
  }
  return 0;
}
