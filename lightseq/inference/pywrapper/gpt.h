#ifdef ENABLE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif
#include "model_base.h"
#include "../model/gpt_encoder.h"
#include "../proto/gpt_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP32;
#endif

#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

namespace lightseq {
namespace cuda {
class Gpt : public LSModel {
 private:
  typedef lightseq::cuda::OperationTypeTraits<gpt_optype> optraits;
  std::shared_ptr<lightseq::cuda::GptEncoder<gpt_optype>> encoder_;

  int* d_input_;
  int* d_sample_id;
  float* d_ppl;
  void* d_buf_;

  int _max_batch_size;
  cudaStream_t stream_;
  cudaStream_t cache_stream_;
  cublasHandle_t hd_;
  lightseq::cuda::GptWeight<gpt_optype> tw_;
  std::set<std::string> available_sampling_methods = {"topk", "topp"};

 public:
  Gpt(const std::string weight_path, const int max_batch_size);

  ~Gpt();

  const int* get_result_ptr();
  const float* get_score_ptr();
  const int get_max_step() { return tw_._max_step; }

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_output_max_shape(int index) override;

#ifdef ENABLE_PYTHON
  py::array_t<float> ppl(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq);

  py::array_t<int> sample(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq,
      std::string sampling_method = "", const int topk = -1,
      const float topp = -1.0f);
#else
  int ppl(const int* input_seq, int batch_size, int batch_seq_len,
          float* result_seq = nullptr);

  std::tuple<int, int> sample(const int* input_seq, int batch_size,
                              int batch_seq_len, int* result_seq = nullptr);
#endif
};

LSMODEL_REGISTER(Gpt);

}  // namespace cuda
}  // namespace lightseq
