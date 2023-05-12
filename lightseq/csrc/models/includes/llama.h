#pragma once
#include "model_base.h"

#include "model_util.h"
#include "llama_weight.h"

#include "launch_llama_emb_layer.h"
#include "llama_layer.h"
#include "linear_layer.h"
#include "rms_norm_layer.h"
#include "generator_layer.h"

namespace lightseq {
class Llama : public LSModel {
 private:
  LlamaWeight<OpType_> tw_;
  std::shared_ptr<Context> _context_ptr;

  LaunchLlamaEmbLayerPtr<OpType_> _launch_llama_emb_layer;
  std::vector<LlamaLayerPtr<OpType_, OpType_>> _llama_layer_vec;
  RMSNormLayerPtr<OpType_, OpType_> _rms_norm_layer;
  LinearLayerPtr<OpType_, OpType_> _linear_layer;
  GeneratorLayerPtr<OpType_> _generator_layer;

  ContextPtr context_ptr;

  Variable* _inp_tokens;  // need to allocate
  Variable* _out_tokens;
  Variable* _pad_mask;

  Variable* _total_caches_k;
  Variable* _total_caches_v;

  int* _llama_out_ptr = nullptr;
  int* _input_ptr = nullptr;
  float* _llama_scores_ptr = nullptr;

  int _max_batch_size;
  GenerateMethod _generate_method;

 public:
  Llama(const std::string weight_path, const int max_batch_size);
  ~Llama();

  void before_forward(int batch_size, int prompt_len, int steps);

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
  void benchmark_mode(bool is_benchmark) override {}
};

LSMODEL_REGISTER(Llama);
}  // namespace lightseq
