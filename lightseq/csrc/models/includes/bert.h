
#include "model_base.h"
#include "layer.h"
#include "transformer_encoder_layer.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType bert_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType bert_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
class Bert : public LSModel {
 private:
 public:
  Bert(const std::string weight_path, const int max_batch_size);

  ~Bert();

  void Infer() override;
  void set_input_ptr(int index, void *input_ptr) override;
  void set_output_ptr(int index, void *output_ptr) override;
  const void *get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(Bert);

}  // namespace lightseq
