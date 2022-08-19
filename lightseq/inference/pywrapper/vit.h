
#include "model_base.h"
#include "../model/vit_encoder.h"
#include "../proto/vit_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType vit_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType vit_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
namespace cuda {
class Vit : public LSModel {
 private:
  typedef OperationTypeTraits<vit_optype> optraits;
  std::shared_ptr<VitEncoder<vit_optype>> encoder_;

  optraits::DataType *d_encoder_output_;
  float *d_input_;
  int *d_padding_mask_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  void *d_buf_;
  VitWeight<vit_optype> tw_;

 public:
  Vit(const std::string weight_path, const int max_batch_size);

  ~Vit();

  void Infer() override;
  void set_input_ptr(int index, void *input_ptr) override;
  void set_output_ptr(int index, void *output_ptr) override;
  const void *get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
  void benchmark_mode(bool is_benchmark) override{};
};

LSMODEL_REGISTER(Vit);

}  // namespace cuda
}  // namespace lightseq
