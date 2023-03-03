#pragma once
#include "model_base.h"

// #include "Gpt_weight.h"
#include "gpt_weight.h"

#include "launch_gpt_emb_layer.h"
#include "gpt_layer.h"
#include "lyr_normalize_layer.h"
#include "linear_layer.h"

#ifdef FP16_MODE
typedef __half OpType_;
#else
typedef float OpType_;
#endif

namespace lightseq {

class Gpt : public LSModel {
 private:
  GptWeight<OpType_> tw_;
  std::shared_ptr<Context> _context_ptr;

  LaunchGptEmbLayerPtr<OpType_> _launch_gpt_emb_layer;
  std::vector<GptLayer<OpType_, OpType_> > _gpt_layers_vec;
  LyrNormalizeLayerPtr<OpType_, OpType_> _lyr_norm_layer;
  GeneratorLayerPtr<OpType_> _generator_layer;

  ContextPtr context_ptr;

  Variable* inp_tokens;  // need to allocate
  Variable* token_emb;
  Variable* pos_emb;

  Variable* Gpt_out;

  int _max_batch_size;
  GenerateMethod _generate_method;

 public:
  Gpt(const std::string weight_path, const int max_batch_size);
  ~Gpt();

  void before_forward(int batch_size, int seq_len);

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(Gpt);

}  // namespace lightseq
