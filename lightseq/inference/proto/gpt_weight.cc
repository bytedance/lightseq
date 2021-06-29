#include "gpt_weight.h"

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
float GptWeight<OperationType::FP32>::float2required(float value) {
  return value;
}

/**
fp16 version, cast fp32 into fp16
*/
template <>
__half GptWeight<OperationType::FP16>::float2required(float value) {
  return __float2half_rn(value);
}

/**
Read model config stored in custom proto file.
*/
template <OperationType OpType_>
void GptWeight<OpType_>::proto_get_model_config(const Gpt &gpt) {
  _hidden_size = gpt.src_embedding().norm_scale_size();
  _inner_size = gpt.encoder_stack()[0].ffn_first_kernel_size() / _hidden_size;
  _max_step = gpt.src_embedding().position_embedding_size() / _hidden_size;
  _src_vocab_size = gpt.src_embedding().token_embedding_size() / _hidden_size;
  _n_enc_layer = gpt.encoder_stack_size();
  _head_num = gpt.model_conf().head_num();
  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;
  _padding_id = gpt.model_conf().src_padding_id();
  if (gpt.model_conf().sampling_method() != "") {
    _sampling_method = gpt.model_conf().sampling_method();
  }
  if (gpt.model_conf().topk() != 0) {
    _topk = gpt.model_conf().topk();
  }
  if (gpt.model_conf().topp() != 0.0) {
    _topp = gpt.model_conf().topp();
  }
  if (gpt.model_conf().eos_id() != 0) {
    _eos_id = gpt.model_conf().eos_id();
  }
}

/**
Load the weights of embedding layer into GPU memory.
*/
template <OperationType OpType_>
std::string GptWeight<OpType_>::proto_parse_emb_wei(
    const GptEmbeddingLayer &layer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  offset.push_back(idx);
  if (layer.token_embedding_size() != _src_vocab_size * _hidden_size)
    return "wrong token_embedding_size !";
  for (float ele : layer.token_embedding()) value.push_back(ele);
  idx += _src_vocab_size * _hidden_size;

  offset.push_back(idx);
  if (layer.position_embedding_size() != _max_step * _hidden_size)
    return "wrong position_embedding_size !";
  for (float ele : layer.position_embedding()) value.push_back(ele);
  idx += _max_step * _hidden_size;

  offset.push_back(idx);
  if (layer.norm_scale_size() != _hidden_size) return "wrong norm_scale_size !";
  for (float ele : layer.norm_scale()) value.push_back(ele);
  idx += _hidden_size;

  offset.push_back(idx);
  if (layer.norm_bias_size() != _hidden_size) return "wrong norm_bias_size !";
  for (float ele : layer.norm_bias()) value.push_back(ele);
  idx += _hidden_size;

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_src_emb_wei = raw_value;
  for (int e : offset)
    _p_d_src_emb_wei.push_back(thrust::raw_pointer_cast(_d_src_emb_wei.data()) +
                               e);

  std::cout << "finish initializing emb_wei from host to device" << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
template <OperationType OpType_>
std::string GptWeight<OpType_>::proto_parse_enc_wei(const Gpt &gpt) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  for (auto enc_layer : gpt.encoder_stack()) {
    offset.push_back(idx);
    if (enc_layer.multihead_norm_scale_size() != _hidden_size)
      return "wrong multihead_norm_scale_size !";
    for (float ele : enc_layer.multihead_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_norm_bias_size() != _hidden_size)
      return "wrong multihead_norm_bias_size !";
    for (float ele : enc_layer.multihead_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_qkv_size() !=
        _hidden_size * _hidden_size * 3)
      return "wrong multihead_project_kernel_qkv_size !";
    for (float ele : enc_layer.multihead_project_kernel_qkv())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
      return "wrong multihead_project_bias_qkv_size !";
    for (float ele : enc_layer.multihead_project_bias_qkv())
      value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "wrong multihead_project_kernel_output_size !";
    for (float ele : enc_layer.multihead_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
      return "wrong multihead_project_bias_output_size !";
    for (float ele : enc_layer.multihead_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_scale_size() != _hidden_size)
      return "wrong ffn_norm_scale_size !";
    for (float ele : enc_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_bias_size() != _hidden_size)
      return "wrong ffn_norm_bias_size !";
    for (float ele : enc_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
      return "wrong ffn_first_kernel_size !";
    for (float ele : enc_layer.ffn_first_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_bias_size() != _inner_size)
      return "wrong ffn_first_bias_size !";
    for (float ele : enc_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
      return "wrong ffn_second_kernel_size !";
    for (float ele : enc_layer.ffn_second_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_bias_size() != _hidden_size)
      return "wrong ffn_second_bias_size !";
    for (float ele : enc_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

  }  // for

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_enc_wei = raw_value;

  for (int e : offset)
    _p_d_enc_wei.push_back(thrust::raw_pointer_cast(_d_enc_wei.data()) + e);
  std::cout << "finish initializing enc_wei from host to device" << std::endl;
  return "";
}

/**
Load the proto file into CPU memory and parse it.
*/
template <OperationType OpType_>
std::string GptWeight<OpType_>::initializing(std::string proto_path) {
  // If weight is of type pb, parse using proto parser.
  if (endswith(weight_path, ".pb")) {
    std::cout << "Parsing protobuf: " << weight_path << std::endl;
    Gpt gpt;
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::fstream raw_input(proto_path, std::ios::in | std::ios::binary);
    if (!gpt.ParseFromIstream(&raw_input)) {
      return "Parse weights from [" + proto_path + "] failed.";
    }

    proto_get_model_config(gpt);

    std::string res = proto_parse_emb_wei(gpt.src_embedding());
    if (!res.empty()) return res;

    res = proto_parse_enc_wei(gpt);
    if (!res.empty()) return res;

    std::cout << "finish initializing all weight from host to device"
              << std::endl;
    // Optional:  Delete all global objects allocated by libprotobuf.
    // google::protobuf::ShutdownProtobufLibrary();
    return "";
  } else if (endswith(weight_path, ".hdf5")) {
    std::cout << "Parsing hdf5: " << weight_path << std::endl;

    hid_t hdf5_file = H5Fopen(weight_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (hdf5_file < 0) {
      return "Unable to read HDF5 file from " + weight_path;
    }
    hdf5_get_model_config(hdf5_file);

    // hdf5_parse_* would throw std::runtime_error on error
    hdf5_parse_emb_wei(hdf5_file);
    hdf5_parse_enc_wei(hdf5_file);
    H5Fclose(hdf5_file);

    std::cout << "Finish loading all weight from host to device" << std::endl;
    return "";
  } else {
    return "Unsupported weight extention for [" + weight_path +
           "]; Supported extensions: .pb, .hdf5\n";
  }
}

template class GptWeight<OperationType::FP16>;
template class GptWeight<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
