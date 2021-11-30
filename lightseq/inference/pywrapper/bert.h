#ifdef ENABLE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif
#include "model_base.h"
#include "../model/bert_encoder.h"
#include "../proto/bert_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType bert_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType bert_optype =
    lightseq::cuda::OperationType::FP32;
#endif

#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

namespace lightseq {
namespace cuda {
class Bert : public LSModel {
 private:
  typedef OperationTypeTraits<bert_optype> optraits;
  std::shared_ptr<BertEncoder<bert_optype>> encoder_;

  optraits::DataType *d_encoder_output_;
  int *d_input_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  void *d_buf_;
  BertWeight<bert_optype> tw_;

  void Infer() override;
  void set_input_ptr(int index, void *input_ptr) override;
  void set_output_ptr(int index, void *output_ptr) override;
  const void *get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;

 public:
  Bert(const std::string weight_path, const int max_batch_size);

  ~Bert();

  const optraits::DataType *get_result_ptr();
  const int get_max_step() { return tw_._max_step; }

#ifdef ENABLE_PYTHON
  py::array_t<float> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq);

#endif
};

LSMODEL_REGISTER(Bert);

}  // namespace cuda
}  // namespace lightseq
