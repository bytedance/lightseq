#include "bert_crf.h"

namespace lightseq {
namespace cuda {

BertCrf::BertCrf(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"encoder_output"}),
      _max_batch_size(max_batch_size) {
  /* --- step.1 initial context --- */
  Context::create_global_context();

  /* --- step.2 load model weights into GPU memory --- */
  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  inp_tokens = new Variable("inp_tokens");
  pad_mask_ptr = cuda_malloc<int>(_max_batch_size * tw_._max_step);
  pad_mask = new Variable("pad_mask", (char *)pad_mask_ptr);

  /* --- step.4 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  // initial LaunchEncEmb layer
  launch_enc_emb_layer.reset(new LaunchEncEmbLayer<OpType_>(
      max_batch_tokens, tw_._padding_id, tw_._hidden_size, tw_._multilg_type));
  launch_enc_emb_layer->load_params(tw_.get_src_emb_wei(), 0);

  // initial TransformerEncoder layers
  float attn_prob_dropout_ratio = 0.0;
  float activation_dropout_ratio = 0.0;
  float hidden_dropout_ratio = 0.0;
  int enc_wei_offset = 0;
  for (int idx = 0; idx < tw_._n_enc_layer; idx++) {
    TransformerEncoderLayerPtr<OpType_, OpType_> enc_layer_(
        new TransformerEncoderLayer<OpType_, OpType_>(
            idx, max_batch_tokens, tw_._max_step, tw_._hidden_size,
            tw_._head_num, tw_._inner_size, attn_prob_dropout_ratio,
            activation_dropout_ratio, hidden_dropout_ratio, true,
            tw_._use_gelu ? "gelu" : "relu", false, tw_._is_post_ln));
    enc_wei_offset +=
        enc_layer_->load_params(tw_.get_enc_wei(), enc_wei_offset);
    enc_layer_vec.push_back(enc_layer_);
  }

  // initial LayerNormalize layer
  lyr_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      max_batch_tokens, tw_._hidden_size));
  lyr_norm_layer->load_params(tw_.get_src_emb_wei(), 2);

  // initial linear layer
  linear_layer.reset(new LinearLayer<OpType_, OpType_>(
      max_batch_tokens, tw_._hidden_size, tw_._num_tags));
  linear_layer->load_params(tw_.get_src_emb_wei(), 4);

  // initial crf layer
  crf_layer.reset(
      new CRFLayer<OpType_>(max_batch_tokens, tw_._hidden_size, tw_._num_tags));
  crf_layer->load_params(tw_.get_src_emb_wei(), 5);

  /* --- step.5 construct network --- */
  Variable *enc_emb = (*launch_enc_emb_layer)(inp_tokens, pad_mask);
  for (auto iter : enc_layer_vec) {
    enc_emb = (*iter)(enc_emb, pad_mask);
  }
  enc_emb = (*lyr_norm_layer)(enc_emb);
  enc_emb = (*linear_layer)(enc_emb);
  bert_out = (*crf_layer)(enc_emb, pad_mask);
}

BertCrf::~BertCrf() { cuda_free(pad_mask_ptr); }

void BertCrf::before_forward(int batch_size, int seq_len) {
  launch_enc_emb_layer->before_forward(batch_size, seq_len);

  for (auto iter : enc_layer_vec) {
    iter->before_forward(batch_size, seq_len);
  }

  lyr_norm_layer->before_forward(batch_size * seq_len);

  linear_layer->before_forward(batch_size, seq_len);
  crf_layer->before_forward(batch_size, seq_len, false, false);
}

void BertCrf::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  before_forward(batch_size, seq_len);

  /* --- notice that the order of forward should be the same with network --- */
  launch_enc_emb_layer->forward();
  for (auto iter : enc_layer_vec) {
    iter->forward();
  }
  lyr_norm_layer->forward();
  linear_layer->forward();
  crf_layer->forward();

  set_output_shape(0, {batch_size, seq_len});
}

void BertCrf::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      inp_tokens->set_value((char *)input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void BertCrf::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      bert_out->set_value((char *)output_ptr);
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *BertCrf::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(bert_out->value());
    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> BertCrf::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> BertCrf::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType BertCrf::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType BertCrf::get_output_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
