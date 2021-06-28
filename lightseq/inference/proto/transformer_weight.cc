#include "transformer_weight.h"

#include "hdf5.h"
#include <fstream>
/**
@file
Load the model weights which stored in custom proto file into GPU memory.
Currently, fp16 and fp32 versions are provided.
Weights in proto file will always be in fp32. For fp16, the weights
  will be casted from fp32 into fp16
*/
namespace lightseq {
namespace cuda {
/**
Cast weights into required datatype.
The datatype of weights in custom proto file will always be in fp32.
*/
template <>
float TransformerWeight<OperationType::FP32>::float2required(float value) {
  return value;
}

/**
fp16 version, cast fp32 into fp16
*/
template <>
__half TransformerWeight<OperationType::FP16>::float2required(float value) {
  return __float2half_rn(value);
}

/**
Read model config stored in custom proto file.
*/
template <OperationType OpType_>
void TransformerWeight<OpType_>::proto_get_model_config(
    const Transformer &transformer, bool only_decoder) {
  _hidden_size = transformer.trg_embedding().norm_scale_size();
  _inner_size =
      transformer.decoder_stack()[0].ffn_first_kernel_size() / _hidden_size;
  _max_step =
      transformer.trg_embedding().position_embedding_size() / _hidden_size;
  if (!only_decoder) {
    _src_vocab_size =
        transformer.src_embedding().token_embedding_size() / _hidden_size;
  }
  _trg_vocab_size =
      transformer.trg_embedding().token_embedding_size() / _hidden_size;
  if (!only_decoder) {
    _n_enc_layer = transformer.encoder_stack_size();
  }
  _n_dec_layer = transformer.decoder_stack_size();
  _head_num = transformer.model_conf().head_num();
  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;
  _weight_per_dec_layer = 18;
  _beam_size = transformer.model_conf().beam_size();
  _extra_decode_length = transformer.model_conf().extra_decode_length();
  _length_penalty = transformer.model_conf().length_penalty();
  _padding_id = transformer.model_conf().src_padding_id();
  _start_id = transformer.model_conf().trg_start_id();
  _end_id = transformer.model_conf().trg_end_id();
  if (_end_id == 0) {
    _end_id = _trg_vocab_size - 1;
  }
  _diverse_lambda = transformer.model_conf().diverse_lambda();
  _sampling_method = transformer.model_conf().sampling_method();
  if (_sampling_method == "") {
    _sampling_method = "beam_search";
  }
  _topk = transformer.model_conf().topk();
  _topp = transformer.model_conf().topp();
  _is_post_ln = transformer.model_conf().is_post_ln();
  _no_scale_embedding = transformer.model_conf().no_scale_embedding();
  _use_gelu = transformer.model_conf().use_gelu();
  _is_multilingual = transformer.model_conf().is_multilingual();
}

/**
Load the weights of embedding layer into GPU memory.
Compared with the encoder, the decoder has more
  encoder output project weights, encoder output project bias,
  logits bias. So we need an "source" parameter to
  distinguish between encoder and decoder
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::proto_parse_emb_wei(
    const EmbeddingLayer &layer, std::string source) {
  int vocab_size = (source == "src") ? _src_vocab_size : _trg_vocab_size;

  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  offset.push_back(idx);
  if (layer.token_embedding_size() != vocab_size * _hidden_size)
    return "Wrong token_embedding_size !";
  for (float ele : layer.token_embedding()) value.push_back(ele);
  idx += vocab_size * _hidden_size;

  offset.push_back(idx);
  if (layer.position_embedding_size() != _max_step * _hidden_size)
    return "Wrong position_embedding_size !";
  for (float ele : layer.position_embedding()) value.push_back(ele);
  idx += _max_step * _hidden_size;

  offset.push_back(idx);
  if (layer.norm_scale_size() != _hidden_size) return "Wrong norm_scale_size !";
  for (float ele : layer.norm_scale()) value.push_back(ele);
  idx += _hidden_size;

  offset.push_back(idx);
  if (layer.norm_bias_size() != _hidden_size) return "Wrong norm_bias_size !";
  for (float ele : layer.norm_bias()) value.push_back(ele);
  idx += _hidden_size;

  if (source == "src") {
    std::vector<_DataType> raw_value;
    for (float e : value) raw_value.push_back(float2required(e));
    _d_src_emb_wei = raw_value;
    for (int e : offset)
      _p_d_src_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_src_emb_wei.data()) + e);
  } else {
    // for trg, encdec_kv_kernel, encdec_kv_bias, logit_bias

    offset.push_back(idx);
    if (layer.encode_output_project_kernel_kv_size() !=
        _hidden_size * _hidden_size * 2 * _n_dec_layer)
      return "Wrong encode_output_project_kernel_kv_size !";
    for (float ele : layer.encode_output_project_kernel_kv())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size * 2 * _n_dec_layer;

    offset.push_back(idx);
    if (layer.encode_output_project_bias_kv_size() !=
        _hidden_size * 2 * _n_dec_layer)
      return "Wrong encode_output_project_bias_kv_size !";
    for (float ele : layer.encode_output_project_bias_kv())
      value.push_back(ele);
    idx += _hidden_size * 2 * _n_dec_layer;

    offset.push_back(idx);
    if (layer.shared_bias_size() != vocab_size)
      return "Wrong shared_bias_size !";
    for (float ele : layer.shared_bias()) value.push_back(ele);
    idx += vocab_size;

    std::vector<_DataType> raw_value;
    for (float e : value) raw_value.push_back(float2required(e));
    _d_trg_emb_wei = raw_value;
    for (int e : offset) {
      _p_d_trg_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_trg_emb_wei.data()) + e);
    }
  }  // trg

  if (_is_multilingual) {
    // fill in language embedding
    std::vector<_DataType> raw_value;
    for (float e : layer.lang_emb()) {
      raw_value.push_back(float2required(e));
    }

    if (source == "src") {
      _d_src_lang_emb = raw_value;
      _p_d_src_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_src_lang_emb.data()));
    } else {
      if (layer.lang_emb_size() / _hidden_size !=
          layer.trg_vocab_mask_size() / _trg_vocab_size) {
        return "Wrong trg_lang_emb_size or trg_vocab_mask_size !";
      }
      _d_trg_lang_emb = raw_value;
      _p_d_trg_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_trg_lang_emb.data()));
      // fill in target vocab mask
      std::vector<int> h_mask;
      for (int ele : layer.trg_vocab_mask()) h_mask.push_back(ele);
      _d_trg_vocab_mask = h_mask;
      _p_d_trg_vocab_mask = thrust::raw_pointer_cast(_d_trg_vocab_mask.data());
    }

    std::cout << "Finish loading multi lingual weights from host to device"
              << std::endl;
  }

  std::cout << "Finish loading " << source << "_emb_wei from host to device"
            << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::proto_parse_enc_wei(
    const Transformer &transformer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  for (auto enc_layer : transformer.encoder_stack()) {
    offset.push_back(idx);
    if (enc_layer.multihead_norm_scale_size() != _hidden_size)
      return "Wrong multihead_norm_scale_size !";
    for (float ele : enc_layer.multihead_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_norm_bias_size() != _hidden_size)
      return "Wrong multihead_norm_bias_size !";
    for (float ele : enc_layer.multihead_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_qkv_size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong multihead_project_kernel_qkv_size !";
    for (float ele : enc_layer.multihead_project_kernel_qkv())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong multihead_project_bias_qkv_size !";
    for (float ele : enc_layer.multihead_project_bias_qkv())
      value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong multihead_project_kernel_output_size !";
    for (float ele : enc_layer.multihead_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
      return "Wrong multihead_project_bias_output_size !";
    for (float ele : enc_layer.multihead_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : enc_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : enc_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (float ele : enc_layer.ffn_first_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : enc_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (float ele : enc_layer.ffn_second_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : enc_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

  }  // for

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_enc_wei = raw_value;

  for (int e : offset)
    _p_d_enc_wei.push_back(thrust::raw_pointer_cast(_d_enc_wei.data()) + e);
  std::cout << "Finish loading enc_wei from host to device" << std::endl;
  return "";
}

/**
Load the weights of decoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::proto_parse_dec_wei(
    const Transformer &transformer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  for (auto dec_layer : transformer.decoder_stack()) {
    offset.push_back(idx);
    if (dec_layer.self_norm_scale_size() != _hidden_size)
      return "Wrong self_norm_scale size !";
    for (float ele : dec_layer.self_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_norm_bias_size() != _hidden_size)
      return "Wrong self_norm_bias_size !";
    for (float ele : dec_layer.self_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_project_kernel_qkv_size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong self_project_kernel_qkv size !";
    for (float ele : dec_layer.self_project_kernel_qkv()) value.push_back(ele);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (dec_layer.self_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong self_project_bias_qkv size !";
    for (float ele : dec_layer.self_project_bias_qkv()) value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (dec_layer.self_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong self_project_kernel_output size !";
    for (float ele : dec_layer.self_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_project_bias_output_size() != _hidden_size)
      return "Wrong self_project_bias_output size !";
    for (float ele : dec_layer.self_project_bias_output()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_scale_size() != _hidden_size)
      return "Wrong encdec_norm_scale size !";
    for (float ele : dec_layer.encdec_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_bias_size() != _hidden_size)
      return "Wrong encdec_norm_bias_size !";
    for (float ele : dec_layer.encdec_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_kernel_q_size() != _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_q size !";
    for (float ele : dec_layer.encdec_project_kernel_q()) value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_q_size() != _hidden_size)
      return "Wrong encdec_project_bias_q size !";
    for (float ele : dec_layer.encdec_project_bias_q()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_output size !";
    for (float ele : dec_layer.encdec_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_output_size() != _hidden_size)
      return "Wrong encdec_project_bias_output size !";
    for (float ele : dec_layer.encdec_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : dec_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : dec_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (float ele : dec_layer.ffn_first_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : dec_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (float ele : dec_layer.ffn_second_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : dec_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

  }  // for

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_dec_wei = raw_value;

  for (int e : offset)
    _p_d_dec_wei.push_back(thrust::raw_pointer_cast(_d_dec_wei.data()) + e);
  std::cout << "Finish loading dec_wei from host to device" << std::endl;
  return "";
}

/**
Read model config stored in custom hdf5 file.
*/
template <OperationType OpType_>
void TransformerWeight<OpType_>::hdf5_get_model_config(hid_t hdf5_file,
                                                       bool only_decoder) {
  _hidden_size = get_hdf5_dataset_size(hdf5_file, "trg_embedding/norm_scale");
  std::cout << "hidden_size " << _hidden_size << std::endl;

  _inner_size =
      get_hdf5_dataset_size(hdf5_file, "decoder_stack/0/ffn_first_kernel") /
      _hidden_size;
  std::cout << "inner_size " << _inner_size << std::endl;

  _max_step =
      get_hdf5_dataset_size(hdf5_file, "trg_embedding/position_embedding") /
      _hidden_size;
  std::cout << "max_step " << _max_step << std::endl;

  if (!only_decoder) {
    _src_vocab_size =
        get_hdf5_dataset_size(hdf5_file, "src_embedding/token_embedding") /
        _hidden_size;
    std::cout << "_src_vocab_size " << _src_vocab_size << std::endl;
  }

  _trg_vocab_size =
      get_hdf5_dataset_size(hdf5_file, "trg_embedding/token_embedding") /
      _hidden_size;
  std::cout << "_trg_vocab_size " << _trg_vocab_size << std::endl;

  if (!only_decoder) {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/n_encoder_stack",
                             H5T_NATIVE_INT, &_n_enc_layer);
    std::cout << "n_enc_layer " << _n_enc_layer << std::endl;
  }

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/n_decoder_stack",
                           H5T_NATIVE_INT, &_n_dec_layer);
  std::cout << "n_dec_layer " << _n_dec_layer << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/head_num", H5T_NATIVE_INT,
                           &_head_num);
  std::cout << "head_num " << _head_num << std::endl;

  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;
  _weight_per_dec_layer = 18;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/beam_size", H5T_NATIVE_INT,
                           &_beam_size);
  std::cout << "_beam_size " << _beam_size << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/extra_decode_length",
                           H5T_NATIVE_INT, &_extra_decode_length);
  std::cout << "_extra_decode_length " << _extra_decode_length << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/length_penalty",
                           H5T_NATIVE_FLOAT, &_length_penalty);
  std::cout << "_length_penalty " << _length_penalty << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/src_padding_id",
                           H5T_NATIVE_INT, &_padding_id);
  std::cout << "_padding_id " << _padding_id << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/trg_start_id", H5T_NATIVE_INT,
                           &_start_id);
  std::cout << "_start_id " << _start_id << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/trg_end_id", H5T_NATIVE_INT,
                           &_end_id);
  std::cout << "_end_id " << _end_id << std::endl;

  if (_end_id == 0) {
    _end_id = _trg_vocab_size - 1;
  }

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/diverse_lambda",
                           H5T_NATIVE_FLOAT, &_diverse_lambda);
  std::cout << "_diverse_lambda " << _diverse_lambda << std::endl;

  char _sampling_method_buf[128];  // get 128 character for sampling method
  int _sampling_method_strlen = read_hdf5_dataset_data(
      hdf5_file, "model_conf/sampling_method", H5T_NATIVE_SCHAR,
      _sampling_method_buf, [](int size) { return size > 128; },
      "Expect model_conf/sampling_method to have less than 128 characters.");
  _sampling_method.assign(_sampling_method_buf, _sampling_method_strlen);
  std::cout << "_sampling_method " << _sampling_method << std::endl;

  if (_sampling_method == "") {
    _sampling_method = "beam_search";
  }

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topk", H5T_NATIVE_INT,
                           &_topk);
  std::cout << "_topk " << _topk << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topp", H5T_NATIVE_FLOAT,
                           &_topp);
  std::cout << "_topp " << _topp << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/is_post_ln", H5T_NATIVE_HBOOL,
                           &_is_post_ln);
  std::cout << "_is_post_ln " << _is_post_ln << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/no_scale_embedding",
                           H5T_NATIVE_HBOOL, &_no_scale_embedding);
  std::cout << "_no_scale_embedding " << _no_scale_embedding << std::endl;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/use_gelu", H5T_NATIVE_HBOOL,
                           &_use_gelu);
  std::cout << "_use_gelu " << _use_gelu << std::endl;

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/is_multilingual",
                             H5T_NATIVE_HBOOL, &_is_multilingual);
  } catch (HDF5DatasetNotFoundError &e) {
    // if this attribute is not found, default initialize it to false
    _is_multilingual = false;
  }
  std::cout << "_is_multilingual " << _is_multilingual << std::endl;
}

/**
Load the weights of embedding layer into GPU memory.
Compared with the encoder, the decoder has more
  encoder output project weights, encoder output project bias,
  logits bias. So we need an "source" parameter to
  distinguish between encoder and decoder
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::hdf5_parse_emb_wei(hid_t hdf5_file,
                                                           std::string source) {
  // int vocab_size = (source == "src") ? _src_vocab_size : _trg_vocab_size;

  // std::vector<int> offset;
  // std::vector<float> value;
  // int idx = 0;

  // offset.push_back(idx);
  // if (layer.token_embedding_size() != vocab_size * _hidden_size)
  //   return "Wrong token_embedding_size !";
  // for (float ele : layer.token_embedding()) value.push_back(ele);
  // idx += vocab_size * _hidden_size;

  // offset.push_back(idx);
  // if (layer.position_embedding_size() != _max_step * _hidden_size)
  //   return "Wrong position_embedding_size !";
  // for (float ele : layer.position_embedding()) value.push_back(ele);
  // idx += _max_step * _hidden_size;

  // offset.push_back(idx);
  // if (layer.norm_scale_size() != _hidden_size) return "Wrong norm_scale_size
  // !"; for (float ele : layer.norm_scale()) value.push_back(ele); idx +=
  // _hidden_size;

  // offset.push_back(idx);
  // if (layer.norm_bias_size() != _hidden_size) return "Wrong norm_bias_size
  // !"; for (float ele : layer.norm_bias()) value.push_back(ele); idx +=
  // _hidden_size;

  // if (source == "src") {
  //   std::vector<_DataType> raw_value;
  //   for (float e : value) raw_value.push_back(float2required(e));
  //   _d_src_emb_wei = raw_value;
  //   for (int e : offset)
  //     _p_d_src_emb_wei.push_back(
  //         thrust::raw_pointer_cast(_d_src_emb_wei.data()) + e);
  // } else {
  //   // for trg, encdec_kv_kernel, encdec_kv_bias, logit_bias

  //   offset.push_back(idx);
  //   if (layer.encode_output_project_kernel_kv_size() !=
  //       _hidden_size * _hidden_size * 2 * _n_dec_layer)
  //     return "Wrong encode_output_project_kernel_kv_size !";
  //   for (float ele : layer.encode_output_project_kernel_kv())
  //     value.push_back(ele);
  //   idx += _hidden_size * _hidden_size * 2 * _n_dec_layer;

  //   offset.push_back(idx);
  //   if (layer.encode_output_project_bias_kv_size() !=
  //       _hidden_size * 2 * _n_dec_layer)
  //     return "Wrong encode_output_project_bias_kv_size !";
  //   for (float ele : layer.encode_output_project_bias_kv())
  //     value.push_back(ele);
  //   idx += _hidden_size * 2 * _n_dec_layer;

  //   offset.push_back(idx);
  //   if (layer.shared_bias_size() != vocab_size)
  //     return "Wrong shared_bias_size !";
  //   for (float ele : layer.shared_bias()) value.push_back(ele);
  //   idx += vocab_size;

  //   std::vector<_DataType> raw_value;
  //   for (float e : value) raw_value.push_back(float2required(e));
  //   _d_trg_emb_wei = raw_value;
  //   for (int e : offset) {
  //     _p_d_trg_emb_wei.push_back(
  //         thrust::raw_pointer_cast(_d_trg_emb_wei.data()) + e);
  //   }
  // }  // trg

  // if (_is_multilingual) {
  //   // fill in language embedding
  //   std::vector<_DataType> raw_value;
  //   for (float e : layer.lang_emb()) {
  //     raw_value.push_back(float2required(e));
  //   }

  //   if (source == "src") {
  //     _d_src_lang_emb = raw_value;
  //     _p_d_src_emb_wei.push_back(
  //         thrust::raw_pointer_cast(_d_src_lang_emb.data()));
  //   } else {
  //     if (layer.lang_emb_size() / _hidden_size !=
  //         layer.trg_vocab_mask_size() / _trg_vocab_size) {
  //       return "Wrong trg_lang_emb_size or trg_vocab_mask_size !";
  //     }
  //     _d_trg_lang_emb = raw_value;
  //     _p_d_trg_emb_wei.push_back(
  //         thrust::raw_pointer_cast(_d_trg_lang_emb.data()));
  //     // fill in target vocab mask
  //     std::vector<int> h_mask;
  //     for (int ele : layer.trg_vocab_mask()) h_mask.push_back(ele);
  //     _d_trg_vocab_mask = h_mask;
  //     _p_d_trg_vocab_mask =
  //     thrust::raw_pointer_cast(_d_trg_vocab_mask.data());
  //   }

  //   std::cout << "Finish loading multi lingual weights from host to device"
  //             << std::endl;
  // }

  // std::cout << "Finish loading " << source << "_emb_wei from host to device"
  //           << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::hdf5_parse_enc_wei(hid_t hdf5_file) {
  // std::vector<int> offset;
  // std::vector<float> value;
  // int idx = 0;

  // for (auto enc_layer : transformer.encoder_stack()) {
  //   offset.push_back(idx);
  //   if (enc_layer.multihead_norm_scale_size() != _hidden_size)
  //     return "Wrong multihead_norm_scale_size !";
  //   for (float ele : enc_layer.multihead_norm_scale()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.multihead_norm_bias_size() != _hidden_size)
  //     return "Wrong multihead_norm_bias_size !";
  //   for (float ele : enc_layer.multihead_norm_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.multihead_project_kernel_qkv_size() !=
  //       _hidden_size * _hidden_size * 3)
  //     return "Wrong multihead_project_kernel_qkv_size !";
  //   for (float ele : enc_layer.multihead_project_kernel_qkv())
  //     value.push_back(ele);
  //   idx += _hidden_size * _hidden_size * 3;

  //   offset.push_back(idx);
  //   if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
  //     return "Wrong multihead_project_bias_qkv_size !";
  //   for (float ele : enc_layer.multihead_project_bias_qkv())
  //     value.push_back(ele);
  //   idx += _hidden_size * 3;

  //   offset.push_back(idx);
  //   if (enc_layer.multihead_project_kernel_output_size() !=
  //       _hidden_size * _hidden_size)
  //     return "Wrong multihead_project_kernel_output_size !";
  //   for (float ele : enc_layer.multihead_project_kernel_output())
  //     value.push_back(ele);
  //   idx += _hidden_size * _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
  //     return "Wrong multihead_project_bias_output_size !";
  //   for (float ele : enc_layer.multihead_project_bias_output())
  //     value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_norm_scale_size() != _hidden_size)
  //     return "Wrong ffn_norm_scale_size !";
  //   for (float ele : enc_layer.ffn_norm_scale()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_norm_bias_size() != _hidden_size)
  //     return "Wrong ffn_norm_bias_size !";
  //   for (float ele : enc_layer.ffn_norm_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
  //     return "Wrong ffn_first_kernel_size !";
  //   for (float ele : enc_layer.ffn_first_kernel()) value.push_back(ele);
  //   idx += _hidden_size * _inner_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_first_bias_size() != _inner_size)
  //     return "Wrong ffn_first_bias_size !";
  //   for (float ele : enc_layer.ffn_first_bias()) value.push_back(ele);
  //   idx += _inner_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
  //     return "Wrong ffn_second_kernel_size !";
  //   for (float ele : enc_layer.ffn_second_kernel()) value.push_back(ele);
  //   idx += _hidden_size * _inner_size;

  //   offset.push_back(idx);
  //   if (enc_layer.ffn_second_bias_size() != _hidden_size)
  //     return "Wrong ffn_second_bias_size !";
  //   for (float ele : enc_layer.ffn_second_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  // }  // for

  // std::vector<_DataType> raw_value;
  // for (float e : value) raw_value.push_back(float2required(e));
  // _d_enc_wei = raw_value;

  // for (int e : offset)
  //   _p_d_enc_wei.push_back(thrust::raw_pointer_cast(_d_enc_wei.data()) + e);
  // std::cout << "Finish loading enc_wei from host to device" << std::endl;
  return "";
}

/**
Load the weights of decoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::hdf5_parse_dec_wei(hid_t hdf5_file) {
  // std::vector<int> offset;
  // std::vector<float> value;
  // int idx = 0;

  // for (auto dec_layer : transformer.decoder_stack()) {
  //   offset.push_back(idx);
  //   if (dec_layer.self_norm_scale_size() != _hidden_size)
  //     return "Wrong self_norm_scale size !";
  //   for (float ele : dec_layer.self_norm_scale()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.self_norm_bias_size() != _hidden_size)
  //     return "Wrong self_norm_bias_size !";
  //   for (float ele : dec_layer.self_norm_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.self_project_kernel_qkv_size() !=
  //       _hidden_size * _hidden_size * 3)
  //     return "Wrong self_project_kernel_qkv size !";
  //   for (float ele : dec_layer.self_project_kernel_qkv())
  //   value.push_back(ele); idx += _hidden_size * _hidden_size * 3;

  //   offset.push_back(idx);
  //   if (dec_layer.self_project_bias_qkv_size() != _hidden_size * 3)
  //     return "Wrong self_project_bias_qkv size !";
  //   for (float ele : dec_layer.self_project_bias_qkv()) value.push_back(ele);
  //   idx += _hidden_size * 3;

  //   offset.push_back(idx);
  //   if (dec_layer.self_project_kernel_output_size() !=
  //       _hidden_size * _hidden_size)
  //     return "Wrong self_project_kernel_output size !";
  //   for (float ele : dec_layer.self_project_kernel_output())
  //     value.push_back(ele);
  //   idx += _hidden_size * _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.self_project_bias_output_size() != _hidden_size)
  //     return "Wrong self_project_bias_output size !";
  //   for (float ele : dec_layer.self_project_bias_output())
  //   value.push_back(ele); idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_norm_scale_size() != _hidden_size)
  //     return "Wrong encdec_norm_scale size !";
  //   for (float ele : dec_layer.encdec_norm_scale()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_norm_bias_size() != _hidden_size)
  //     return "Wrong encdec_norm_bias_size !";
  //   for (float ele : dec_layer.encdec_norm_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_project_kernel_q_size() != _hidden_size *
  //   _hidden_size)
  //     return "Wrong encdec_project_kernel_q size !";
  //   for (float ele : dec_layer.encdec_project_kernel_q())
  //   value.push_back(ele); idx += _hidden_size * _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_project_bias_q_size() != _hidden_size)
  //     return "Wrong encdec_project_bias_q size !";
  //   for (float ele : dec_layer.encdec_project_bias_q()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_project_kernel_output_size() !=
  //       _hidden_size * _hidden_size)
  //     return "Wrong encdec_project_kernel_output size !";
  //   for (float ele : dec_layer.encdec_project_kernel_output())
  //     value.push_back(ele);
  //   idx += _hidden_size * _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.encdec_project_bias_output_size() != _hidden_size)
  //     return "Wrong encdec_project_bias_output size !";
  //   for (float ele : dec_layer.encdec_project_bias_output())
  //     value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_norm_scale_size() != _hidden_size)
  //     return "Wrong ffn_norm_scale_size !";
  //   for (float ele : dec_layer.ffn_norm_scale()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_norm_bias_size() != _hidden_size)
  //     return "Wrong ffn_norm_bias_size !";
  //   for (float ele : dec_layer.ffn_norm_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
  //     return "Wrong ffn_first_kernel_size !";
  //   for (float ele : dec_layer.ffn_first_kernel()) value.push_back(ele);
  //   idx += _hidden_size * _inner_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_first_bias_size() != _inner_size)
  //     return "Wrong ffn_first_bias_size !";
  //   for (float ele : dec_layer.ffn_first_bias()) value.push_back(ele);
  //   idx += _inner_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
  //     return "Wrong ffn_second_kernel_size !";
  //   for (float ele : dec_layer.ffn_second_kernel()) value.push_back(ele);
  //   idx += _hidden_size * _inner_size;

  //   offset.push_back(idx);
  //   if (dec_layer.ffn_second_bias_size() != _hidden_size)
  //     return "Wrong ffn_second_bias_size !";
  //   for (float ele : dec_layer.ffn_second_bias()) value.push_back(ele);
  //   idx += _hidden_size;

  // }  // for

  // std::vector<_DataType> raw_value;
  // for (float e : value) raw_value.push_back(float2required(e));
  // _d_dec_wei = raw_value;

  // for (int e : offset)
  //   _p_d_dec_wei.push_back(thrust::raw_pointer_cast(_d_dec_wei.data()) + e);
  // std::cout << "Finish loading dec_wei from host to device" << std::endl;
  return "";
}

/**
Load the proto file into CPU memory and parse it.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::initializing(std::string weight_path,
                                                     bool only_decoder) {
  // If weight is of type pb, parse using proto parser.
  if (endswith(weight_path, ".pb")) {
    std::cout << "Parsing protobuf: " << weight_path << std::endl;
    Transformer transformer;
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::fstream raw_input(weight_path, std::ios::in | std::ios::binary);
    if (!transformer.ParseFromIstream(&raw_input)) {
      return "Parse weights from [" + weight_path + "] failed.";
    }
    proto_get_model_config(transformer, only_decoder);

    if (_hidden_size % 4 != 0) {
      return "hidden_size should be a multiple of 4 to avoid misaligned "
             "address "
             "in CUDA";
    }

    std::string res;
    if (!only_decoder) {
      res = proto_parse_emb_wei(transformer.src_embedding(), "src");
      if (!res.empty()) return res;
    }

    res = proto_parse_emb_wei(transformer.trg_embedding(), "trg");
    if (!res.empty()) return res;

    if (!only_decoder) {
      res = proto_parse_enc_wei(transformer);
      if (!res.empty()) return res;
    }

    res = proto_parse_dec_wei(transformer);
    if (!res.empty()) return res;

    std::cout << "Finish loading all weight from host to device" << std::endl;
    // Optional:  Delete all global objects allocated by libprotobuf.
    // google::protobuf::ShutdownProtobufLibrary();
    return "";
  } else if (endswith(weight_path, ".hdf5")) {
    std::cout << "Parsing hdf5: " << weight_path << std::endl;

    hid_t hdf5_file = H5Fopen(weight_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hdf5_get_model_config(hdf5_file, only_decoder);
    H5Fclose(hdf5_file);

    return "Debugging abort";
  } else {
    return "Unsupported weight extention for [" + weight_path +
           "]; Supported extensions: .pb, .hdf5\n";
  }
}

template class TransformerWeight<OperationType::FP16>;
template class TransformerWeight<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
