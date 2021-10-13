#ifdef ENABLE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

#include "../model/decoder.h"
#include "../model/encoder.h"
#include "../proto/transformer_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType transformer_optytpe =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType transformer_optytpe =
    lightseq::cuda::OperationType::FP32;
#endif

#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

namespace lightseq {
namespace cuda {
class Transformer {
 private:
  typedef OperationTypeTraits<transformer_optytpe> optraits;
  std::shared_ptr<Encoder<transformer_optytpe>> encoder_;
  std::shared_ptr<Decoder<transformer_optytpe>> decoder_;

  optraits::DataType *d_encoder_output_;
  int *d_input_;
  int *d_output_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  TransformerWeight<transformer_optytpe> tw_;

  int get_output_seq_len();

 public:
  Transformer(const std::string weight_path, const int max_batch_size);

  ~Transformer();

  const int *get_result_ptr();
  const float *get_score_ptr();

#ifdef ENABLE_PYTHON
  std::tuple<py::array_t<int>, py::array_t<float>> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
      bool multiple_output = false);
#else
  std::tuple<int, int, int> infer(int *input_seq, int batch_size,
                                  int batch_seq_len, int *result_seq = nullptr,
                                  float *scores = nullptr,
                                  bool multiple_output = false);
#endif
};
}  // namespace cuda
}  // namespace lightseq
