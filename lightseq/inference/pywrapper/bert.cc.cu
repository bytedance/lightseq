#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../model/bert_encoder.h"
#include "../proto/bert_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType bert_optytpe =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType bert_optytpe =
    lightseq::cuda::OperationType::FP32;
#endif

namespace py = pybind11;

namespace lightseq {
namespace cuda {
class Bert {
 private:
  typedef OperationTypeTraits<bert_optytpe> optraits;
  BertEncoder<bert_optytpe> *bert;

  optraits::DataType *d_encoder_output_;
  int *d_input_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  void *d_buf_;
  BertWeight<bert_optytpe> tw_;

 public:
  Bert(const std::string weight_path, const int max_batch_size) {
    /* ---step1. init environment--- */
    _max_batch_size = max_batch_size;
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

    bert =
        new BertEncoder<bert_optytpe>(max_batch_size, d_input_, d_padding_mask_,
                                      d_encoder_output_, tw_, stream_, hd_);
    res = bert->check();
    if (!res.empty()) {
      throw std::runtime_error(res);
    }

    long buf_bytesize = bert->compute_buffer_bytesize();
    std::cout << "Bert buf_bytesize: " << buf_bytesize << std::endl;

    // encoder and decoder use the same buffer to save gpu memory useage
    CHECK_GPU_ERROR(cudaMalloc(&d_buf_, (size_t)buf_bytesize));
    bert->init_buffer(d_buf_);
    CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));
  }

  ~Bert() {
    CHECK_GPU_ERROR(cudaFree(d_input_));
    CHECK_GPU_ERROR(cudaFree(d_padding_mask_));
    CHECK_GPU_ERROR(cudaFree(d_encoder_output_));
    CHECK_GPU_ERROR(cudaFree(d_buf_));
  }

  py::array_t<float> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
      py::array_t<int, py::array::c_style | py::array::forcecast> attn_mask) {
    // deal with input
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int *input_seq_data = input_seq_out.data(0, 0);
    const int *attn_mask_data = attn_mask.data(0, 0);

    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);
    if (batch_size > _max_batch_size) {
      throw std::runtime_error(
          "batch size of input greater than max_batch_size");
    }
    if (batch_seq_len > tw_._max_step) {
      throw std::runtime_error("seq len of input greater than max_step");
    }
    CHECK_GPU_ERROR(cudaMemcpyAsync(d_input_, input_seq_data,
                                    sizeof(int) * input_seq_out.size(),
                                    cudaMemcpyHostToDevice, stream_));
    CHECK_GPU_ERROR(cudaMemcpyAsync(d_padding_mask_, attn_mask_data,
                                    sizeof(int) * attn_mask.size(),
                                    cudaMemcpyHostToDevice, stream_));

    // Start inference and copy result
    bert->run_one_infer(batch_size, batch_seq_len);

    auto bert_output =
        py::array_t<float>({batch_size, batch_seq_len, tw_._hidden_size});
    float *bert_output_data = bert_output.mutable_data(0, 0, 0);
    std::vector<optraits::DataType> h_bert_out(bert_output.size());

    CHECK_GPU_ERROR(
        cudaMemcpyAsync(h_bert_out.data(), d_encoder_output_,
                        sizeof(optraits::DataType) * bert_output.size(),
                        cudaMemcpyDeviceToHost, stream_));
    CHECK_GPU_ERROR(cudaStreamSynchronize(stream_));

    for (auto i = 0; i < h_bert_out.size(); i++) {
      float data;
      if (bert_optytpe == OperationType::FP16) {
        data = __half2float(h_bert_out[i]);
      } else {
        data = h_bert_out[i];
      }
      bert_output_data[i] = data;
    }
    return bert_output;
  };
};
}  // namespace cuda
}  // namespace lightseq
