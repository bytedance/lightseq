#include "encdec_kv_layer.h"

namespace lightseq {

template <typename T1, typename T2>
EncDecKvLayer<T1, T2>::EncDecKvLayer(int nshared_layer, int layer_id, int max_batch_tokens, int hidden_size, int num_heads)
    : Layer("EncDecKvLayer"),  // necessary
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _nshared_layer(nshared_layer),
      // operators
      _kv_linear(new FeedForwardOp<T1, T2>(max_batch_tokens, 2 * hidden_size,
                                            hidden_size)),
      _bias_add_transform_20314(new BiasAddTrans20314<T1, T2>(
          max_batch_tokens, num_heads, hidden_size, 2)){
  // parameters
  _enc_kvw = new Variable("_enc_kvw");
  _enc_kvb = new Variable("_enc_kvb");

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
std::tuple<Variable*, Variable*> EncDecKvLayer<T1, T2>::operator()(Variable* enc_out) {
  LAYER_PRE_INPUTS({enc_out});

  Variable* kv_out = (*_kv_linear)(enc_out, _enc_kvw);

  std::tuple<Variable*, Variable*, Variable*> transform_20314_out =
      (*_bias_add_transform_20314)(kv_out, _enc_kvb);
  Variable* k_out = std::get<0>(transform_20314_out);
  Variable* v_out = std::get<1>(transform_20314_out);
  
  LAYER_POST_OUTPUTS({k_out, v_out});
  
  return std::make_tuple(k_out, v_out);
}

template <typename T1, typename T2>
void EncDecKvLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  _batch_tokens = batch_size * seq_len;

  _kv_linear->before_forward(_batch_tokens);

  _bias_add_transform_20314->before_forward(batch_size, seq_len);
}

template <typename T1, typename T2>
void EncDecKvLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int EncDecKvLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  int offset = 0;
  _enc_kvw->set_value((char*)(para_ptr + offset));
  _enc_kvw->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * _hidden_size * 2;

  _enc_kvb->set_value((char*)(para_ptr + offset));
  _enc_kvb->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * 2;


  return offset;
}

template <typename T1, typename T2>
int EncDecKvLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;
  _enc_kvw->set_value((char*)para_vec[offset + size]), size++;
  _enc_kvb->set_value((char*)para_vec[offset + size]), size++;
  return size;
}

}  // namespace lightseq
