#pragma once
#include "launch_llama_emb.h"
#include "layer.h"

namespace lightseq {

template <typename T>
class LaunchLlamaEmbLayer : public Layer {
 private:
  // operators
  LaunchLlamaEmbOp<T>* _launch_llama_op = nullptr;

  // parameters
  Variable* _token_emb;

 public:
  LaunchLlamaEmbLayer(int max_batch_tokens, int max_step, int max_batch_size,
                    int beam_size, int pad_id, int hidden_dim)
      : Layer("LaunchLlamaEmbLayer"),
        _launch_llama_op(new LaunchLlamaEmbOp<T>(max_batch_tokens, max_step,
                                             max_batch_size, beam_size, pad_id,
                                             hidden_dim)) {
    _token_emb = new Variable("token_emb", g_dtype<T>());

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LaunchLlamaEmbLayer() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* inp) {
    set_inputs({inp});

    std::tuple<Variable*, Variable*, Variable*> out =
        (*_launch_llama_op)(inp, _token_emb);

    set_outputs({std::get<0>(out), std::get<1>(out), std::get<2>(out)});
    return out;
  }

  void before_forward(int batch_size, int seq_len, int offset) {
    _launch_llama_op->before_forward(batch_size, seq_len, offset);
  }

  void before_backward() {}

  int load_params(const std::vector<const T*>& para_vec, int offset) {
    _token_emb->set_value((char*)para_vec[offset]);
    return 0;
  }
};

template class LaunchLlamaEmbLayer<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchLlamaEmbLayer<__half>;
#endif

template <class T>
using LaunchLlamaEmbLayerPtr = std::shared_ptr<LaunchLlamaEmbLayer<T>>;

}  // namespace lightseq
