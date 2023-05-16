#include "big_gpt_weight.h"

#include <fstream>

/**
@file
Load the model weights which stored in custom proto file into GPU memory.
Currently, fp16 and fp32 versions are provided.
Weights in proto file will always be in fp32. For fp16, the weights
  will be casted from fp32 into fp16
*/

namespace lightseq {

/**
Cast weights into required datatype.
The datatype of weights in custom proto file will always be in fp32.
*/
template <>
float BigGptWeight<float>::float2required(float value) {
  return value;
}

#ifdef LIGHTSEQ_cuda
/**
fp16 version, cast fp32 into fp16
*/
template <>
__half BigGptWeight<__half>::float2required(float value) {
  return __float2half_rn(value);
}
#endif

/**
Read model config stored in custom hdf5 file.
*/
template <typename T>
void BigGptWeight<T>::hdf5_get_model_config(hid_t hdf5_file) {
  _hidden_size = get_hdf5_dataset_size(hdf5_file, "src_embedding/norm_scale");

  _inner_size =
      get_hdf5_dataset_size(hdf5_file, "encoder_stack/0/ffn_first_kernel") /
      _hidden_size;

  _max_step =
      get_hdf5_dataset_size(hdf5_file, "src_embedding/position_embedding") /
      _hidden_size;

  _src_vocab_size =
      get_hdf5_dataset_size(hdf5_file, "src_embedding/token_embedding") /
      _hidden_size;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/n_encoder_stack",
                           H5T_NATIVE_INT, &_n_enc_layer);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/head_num", H5T_NATIVE_INT,
                           &_head_num);

  _dim_per_head = _hidden_size / _head_num;

  _weight_per_enc_layer = 12;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/src_padding_id",
                           H5T_NATIVE_INT, &_padding_id);

  // special handling for string reading
  // string were converted to numpy array of np.int8 in python
  // hence needed to be read as an char array here
  char _sampling_method_buf[128];  // get 128 character for sampling method
  int _sampling_method_strlen = read_hdf5_dataset_data(
      hdf5_file, "model_conf/sampling_method", H5T_NATIVE_CHAR,
      _sampling_method_buf, [](int size) { return size > 128; },
      "Expect model_conf/sampling_method to have less than 128 characters.");
  std::string _sampling_method_read =
      std::string(_sampling_method_buf, _sampling_method_strlen);
  if (_sampling_method_read != "") {
    _sampling_method = _sampling_method_read;
  }

  int _extra_decode_length_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/extra_decode_length",
                           H5T_NATIVE_INT, &_extra_decode_length_read);
  if (_extra_decode_length_read > 0) {
    _extra_decode_length = _extra_decode_length_read;
  } else {
    _extra_decode_length = _max_step;
  }

  int _topk_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topk", H5T_NATIVE_INT,
                           &_topk_read);
  if (_topk_read != 0) {
    _topk = _topk_read;
  }

  float _topp_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topp", H5T_NATIVE_FLOAT,
                           &_topp_read);
  if (_topp_read != 0.0) {
    _topp = _topp_read;
  }

  int _eos_id_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/eos_id", H5T_NATIVE_INT,
                           &_eos_id_read);
  if (_eos_id_read != 0) {
    _eos_id = _eos_id_read;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/beam_size", H5T_NATIVE_INT,
                             &_beam_size);
  } catch (HDF5DatasetNotFoundError &e) {
    _beam_size = 1;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/length_penalty",
                             H5T_NATIVE_FLOAT, &_length_penalty);
  } catch (HDF5DatasetNotFoundError &e) {
    _length_penalty = 1.0;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/diverse_lambda",
                             H5T_NATIVE_FLOAT, &_diverse_lambda);
  } catch (HDF5DatasetNotFoundError &e) {
    _diverse_lambda = 0.;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/use_gelu", H5T_NATIVE_HBOOL,
                             &_use_gelu);
  } catch (HDF5DatasetNotFoundError &e) {
    _use_gelu = true;
  }
}

/**
Load the weights of embedding layer into GPU memory.
*/
template <typename T>
void BigGptWeight<T>::hdf5_parse_emb_wei(hid_t hdf5_file) {
  std::string dataset_prefix = "src_embedding";
  size_t value_size = _src_vocab_size * _hidden_size +
                      _max_step * _hidden_size + _hidden_size * 2;

  std::vector<int> offset;
  std::vector<float> value(value_size);  // preallocate vector for performance
  std::cout << "loading " << value_size * sizeof(T) / (1024 * 1024)
            << " MB of embedding weight." << std::endl;
  int idx = 0;

  const size_t max_buffer_size = std::max(_src_vocab_size, _max_step) * _hidden_size;
  float* source_buffer;
  T* target_buffer;
  cudaMalloc(&source_buffer, max_buffer_size * sizeof(float));
  cudaMalloc(&target_buffer, max_buffer_size * sizeof(T));
  T* addr = nullptr;
  size_t buffer_size;

  
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/token_embedding", H5T_NATIVE_FLOAT,
      value.data() + idx,
      [=](int size) { return size != _src_vocab_size * _hidden_size; },
      "Wrong token_embedding_size !");
  buffer_size = _src_vocab_size * _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);


  
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/position_embedding", H5T_NATIVE_FLOAT,
      value.data() + idx,
      [=](int size) { return size != _max_step * _hidden_size; },
      "Wrong position_embedding_size !");
  buffer_size = _max_step * _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/norm_scale", H5T_NATIVE_FLOAT,
      value.data() + idx, [=](int size) { return size != _hidden_size; },
      "Wrong norm_scale_size !");
  buffer_size = _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/norm_bias", H5T_NATIVE_FLOAT,
      value.data() + idx, [=](int size) { return size != _hidden_size; },
      "Wrong norm_bias_size !");
  buffer_size = _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  value.clear();
  value.shrink_to_fit();
  cudaFree(source_buffer);
  cudaFree(target_buffer);

  std::cout << "finish initializing emb_wei from host to device" << std::endl;
}

/**
Load the weights of encoder into GPU memory.
*/
template <typename T>
void BigGptWeight<T>::hdf5_parse_enc_wei(hid_t hdf5_file) {
  size_t value_size =
      (_hidden_size * 2 + _hidden_size * _hidden_size * 3 + _hidden_size * 3 +
       _hidden_size * _hidden_size + _hidden_size * 3 +
       _hidden_size * _inner_size + _inner_size + _hidden_size * _inner_size +
       _hidden_size) *
      _n_enc_layer;
  std::vector<int> offset;
  std::vector<float> value(value_size);
  std::cout << "loading " << value_size * sizeof(T) / (1024 * 1024)
            << " MB of encoder weight." << std::endl;


  std::vector<size_t> value_size_vec =
      {_hidden_size * 2 , _hidden_size * _hidden_size * 3 , _hidden_size * 3 ,
       _hidden_size * _hidden_size , _hidden_size * 3 ,
       _hidden_size * _inner_size , _inner_size , _hidden_size * _inner_size ,
       _hidden_size};
  size_t max_value_size =  *max_element(value_size_vec.begin(), value_size_vec.end());


  const size_t max_buffer_size = max_value_size;
  float* source_buffer;
  T* target_buffer;
  cudaMalloc(&source_buffer, max_buffer_size * sizeof(float));
  cudaMalloc(&target_buffer, max_buffer_size * sizeof(T));
  T* addr = nullptr;
  size_t buffer_size;


  int idx = 0;
  for (int layer_id = 0; layer_id < _n_enc_layer; ++layer_id) {
    std::string dataset_prefix = "encoder_stack/" + std::to_string(layer_id);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_norm_scale", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong multihead_norm_scale_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_norm_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong multihead_norm_bias_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_qkv",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size * _hidden_size * 3; },
        "Wrong multihead_project_kernel_qkv_size !");
    buffer_size = _hidden_size * _hidden_size * 3;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_bias_qkv",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size * 3; },
        "Wrong multihead_project_bias_qkv_size !");
    buffer_size = _hidden_size * 3;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_output",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size * _hidden_size; },
        "Wrong multihead_project_kernel_output_size !");
    buffer_size = _hidden_size * _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_bias_output",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size; },
        "Wrong multihead_project_bias_output_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_norm_scale", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_norm_scale_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_norm_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_norm_bias_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_first_kernel", H5T_NATIVE_FLOAT,
        value.data() + idx,
        [=](int size) { return size != _hidden_size * _inner_size; },
        "Wrong ffn_first_kernel_size !");
    buffer_size = _hidden_size * _inner_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_first_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _inner_size; },
        "Wrong ffn_first_bias_size !");
    buffer_size = _inner_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_second_kernel", H5T_NATIVE_FLOAT,
        value.data() + idx,
        [=](int size) { return size != _hidden_size * _inner_size; },
        "Wrong ffn_second_kernel_size !");
    buffer_size = _hidden_size * _inner_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_second_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_second_bias_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_src_emb_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);
  }  // for

  value.clear();
  value.shrink_to_fit();
  cudaFree(source_buffer);
  cudaFree(target_buffer);
}

/**
Load the proto file into CPU memory and parse it.
*/
template <typename T>
std::string BigGptWeight<T>::initializing(std::string weight_path) {
  cudaStreamCreate(&stream);
  // If weight is of type pb, parse using proto parser.
  if (endswith(weight_path, ".hdf5")) {
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
    cudaStreamSynchronize(stream);

    std::cout << "Finish loading all weight from host to device" << std::endl;
    return "";
  } else {
    return "Unsupported weight extention for [" + weight_path +
           "]; Supported extensions: .pb, .hdf5\n";
  }
}
#ifdef LIGHTSEQ_cuda
template class BigGptWeight<__half>;
#endif
template class BigGptWeight<float>;

}  // namespace lightseq
