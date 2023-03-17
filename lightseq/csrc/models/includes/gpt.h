#pragma once
#include "model_base.h"

#include "model_util.h"
#include "gpt_weight.h"
#include "launch_gpt_emb_layer.h"
#include "gpt_layer.h"
#include "lyr_normalize_layer.h"
#include "linear_layer.h"
#include "generator_layer.h"

namespace lightseq {

class Gpt : public LSModel {
 private:
  GptWeight<OpType_> tw_;
  std::shared_ptr<Context> _context_ptr;

  LaunchGptEmbLayerPtr<OpType_> _launch_gpt_emb_layer;
  std::vector<GptLayerPtr<OpType_, OpType_> > _gpt_layers_vec;
  LyrNormalizeLayerPtr<OpType_, OpType_> _lyr_norm_layer;
  LinearLayerPtr<OpType_, OpType_> _linear_layer;
  GeneratorLayerPtr<OpType_> _generator_layer;

  ContextPtr context_ptr;

  Variable* _inp_tokens;  // need to allocate
  Variable* _out_tokens;
  Variable* _out_scores;

  Variable* _total_caches_k;
  Variable* _total_caches_v;

  int* _gpt_out_ptr = nullptr;
  int* _input_ptr = nullptr;

  int _max_batch_size;
  GenerateMethod _generate_method;

 public:
  Gpt(const std::string weight_path, const int max_batch_size);
  ~Gpt();

  void before_forward(int batch_size, int prompt_len, int steps);

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
