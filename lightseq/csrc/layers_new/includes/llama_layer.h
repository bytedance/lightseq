#pragma once
#include "layer.h"
#include "llama_attention_layer.h"
#include "llama_mlp_layer.h"

namespace lightseq {

template <class T1, class T2>
class LlamaLayer : public Layer {
 private:
  LlamaAttentionLayerPtr<T1, T2> _attn_layer;
  LlamaMLPLayerPtr<T1, T2> _mlp_layer;

  int _layer_id;

 public:
  LlamaLayer(int max_batch_size, int max_seq_len, int hidden_size, int inner_dim,
            int num_heads, int beam_size);
  virtual ~LlamaLayer() {}

  Variable* operator()(Variable* inp, Variable* cache_k, Variable* cache_v,
                       Variable* pad_mask);

  void before_forward(int batch_size, int seq_len, int prompt_len) {
    
    std::cout << "step.1-2-1\n" << std::endl;
    _attn_layer->before_forward(batch_size, seq_len, prompt_len);
    std::cout << "step.1-2-2\n" << std::endl;
    _mlp_layer->before_forward(batch_size, seq_len);
    std::cout << "step.1-2-3\n" << std::endl;
  }

  size_t load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class LlamaLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class LlamaLayer<__half, __half>;
#endif

template <class T1, class T2>
using LlamaLayerPtr = std::shared_ptr<LlamaLayer<T1, T2>>;

}  // namespace lightseq
