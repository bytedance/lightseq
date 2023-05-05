#include "llama_mlp_layer.h"

namespace lightseq {
namespace cuda {

template <typename T1, typename T2>
LlamaMLPLayer<T1, T2>::LlamaMLPLayer(int max_batch_tokens, int hidden_dim,
                                     int inner_dim)
    : Layer("LlamaMLPLayer"),
      _max_batch_tokens(max_batch_tokens),
      _hidden_dim(hidden_dim),
      _inner_dim(inner_dim),
      _gate_linear(
          new LinearOp<T1, T2>(max_batch_tokens, inner_dim, hidden_dim)),
      _up_linear(new LinearOp<T1, T2>(max_batch_tokens, inner_dim, hidden_dim)),
      _down_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_dim, inner_dim)) {
  _gate_linear_weight =
      new Variable("_gate_linear_weight", g_dtype<T1>(), g_dtype<T2>());
  _up_linear_weight =
      new Variable("_up_linear_weight", g_dtype<T1>(), g_dtype<T2>());
  _down_linear_weight =
      new Variable("_down_linear_weight", g_dtype<T1>(), g_dtype<T2>());
}

}  // namespace cuda
}  // namespace lightseq
