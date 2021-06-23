#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../model/decoder.h"
#include "../model/encoder.h"
#include "../proto/transformer_weight.h"
#include "../tools/util.h"

namespace py = pybind11;

#ifdef FP16_MODE
const lightseq::cuda::OperationType decoder_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType decoder_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
namespace cuda {
class TransformerDecoder {
 private:
  typedef lightseq::cuda::OperationTypeTraits<decoder_optype> optraits;
  lightseq::cuda::Decoder<decoder_optype> *decoder_;

  optraits::DataType *d_encoder_output_;
  int *d_output_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  lightseq::cuda::TransformerWeight<decoder_optype> tw_;

 public:
  TransformerDecoder(const std::string weight_path, const int max_batch_size)
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
    std::string res = tw_.initializing(model_weights_path, true);
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

    // instantiate encoder
    // FIXME: padding mask should be passed from user
    // thrust::device_vector<int> d_padding_mask_ =
    //     std::vector<int>(_max_batch_size * tw_._max_step, 0);
    lightseq::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_padding_mask_, _max_batch_size * tw_._max_step * sizeof(int)));

    lightseq::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_encoder_output_, _max_batch_size * tw_._max_step * tw_._hidden_size *
                                sizeof(optraits::DataType)));

    lightseq::cuda::CHECK_GPU_ERROR(cudaMalloc(
        &d_output_,
        _max_batch_size * tw_._beam_size * tw_._max_step * sizeof(int)));

    decoder_ = new lightseq::cuda::Decoder<decoder_optype>(
        _max_batch_size, d_padding_mask_, d_encoder_output_, d_output_, tw_,
        stream_, hd_, true);
    res = decoder_->check();
    if (!res.empty()) {
      throw std::runtime_error(res);
    }

    long buf_bytesize = decoder_->compute_buffer_bytesize();
    std::cout << "decoder buf_bytesize: " << buf_bytesize << std::endl;

    void *d_buf_;
    // encoder and decoder use the same buffer to save gpu memory useage
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMalloc((void **)&d_buf_, (size_t)buf_bytesize));
    decoder_->init_buffer(d_buf_);
    cuerr = cudaStreamSynchronize(stream_);
    if (cuerr != cudaSuccess) {
      std::cout << "failed to init GPU for transformer: " << std::endl;
      std::runtime_error(std::string(cudaGetErrorString(cuerr)));
    }
  }

  py::array_t<int> infer(
      py::array_t<float, py::array::c_style | py::array::forcecast>
          encoder_output,
      py::array_t<int, py::array::c_style | py::array::forcecast>
          encoder_mask) {
    auto encoder_out = encoder_output.mutable_unchecked<3>();
    auto encoder_mask_out = encoder_mask.mutable_unchecked<2>();
    const float *encoder_output_data = encoder_out.data(0, 0, 0);
    const int *encoder_mask_data = encoder_mask_out.data(0, 0);
    std::vector<optraits::DataType> h_encoder_out(encoder_out.size());
    for (auto i = 0; i < encoder_out.size(); i++) {
      optraits::DataType data;
      if (decoder_optype == lightseq::cuda::OperationType::FP16) {
        data = __float2half_rn(encoder_output_data[i]);
      } else {
        data = encoder_output_data[i];
      }
      h_encoder_out[i] = data;
    }

    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMemcpyAsync(d_encoder_output_, h_encoder_out.data(),
                        sizeof(optraits::DataType) * encoder_out.size(),
                        cudaMemcpyHostToDevice, stream_));
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMemcpyAsync(d_padding_mask_, encoder_mask_data,
                        sizeof(int) * encoder_mask_out.size(),
                        cudaMemcpyHostToDevice, stream_));

    int batch_size = encoder_out.shape(0);
    int batch_seq_len = encoder_out.shape(1);
    decoder_->run_one_infer(batch_size, batch_seq_len);
    int tokens_size = decoder_->_cur_step + 1;
    int beam_size = tw_._beam_size;
    auto tokens = py::array_t<int>({batch_size, beam_size, tokens_size});
    int *tokens_data = tokens.mutable_data(0, 0);
    lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_output_,
                                               sizeof(int) * tokens.size(),
                                               cudaMemcpyDeviceToHost));
    return tokens;
  }
};
}  // namespace cuda
}  // namespace lightseq
