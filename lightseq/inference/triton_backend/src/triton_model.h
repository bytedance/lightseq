
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "bert.h"
#include "gpt.h"
#include "transformer.h"
#include "model_base.h"
#include "quant_transformer.h"

namespace triton {
namespace backend {
namespace lightseq {
/////////////
//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 private:
  std::unordered_map<std::string, TRITONSERVER_DataType> input_data_type_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_data_type_map_;

 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);
  virtual ~ModelState() = default;

  const std::string& ModelFileName() const { return file_name_; }

  // Datatype of the input and output tensor
  TRITONSERVER_DataType GetInputDataTypeByName(std::string input_name) {
    std::unordered_map<std::string, TRITONSERVER_DataType>::iterator iter =
        input_data_type_map_.find(input_name);
    if (iter == input_data_type_map_.end()) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  "input_name error, cannot found in input_data_type_map_");
    }
    return iter->second;
  }
  TRITONSERVER_DataType GetOutputDataTypeByName(std::string output_name) {
    std::unordered_map<std::string, TRITONSERVER_DataType>::iterator iter =
        output_data_type_map_.find(output_name);
    if (iter == output_data_type_map_.end()) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  "output_name error, cannot found in output_data_type_map_");
    }
    return iter->second;
  }

  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_; }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  std::string GetModelType() { return model_type_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  std::string file_name_;

  bool shape_initialized_;
  std::vector<int64_t> nb_shape_;
  std::vector<int64_t> shape_;

  std::string model_type_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false) {
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "ModelState Created success");

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::TensorShape(std::vector<int64_t>& shape) {
  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be the shape from the model configuration. For batch models
  // the output shape will be the shape from the model configuration
  // prepended with [ -1 ] to represent the batch dimension. The
  // backend "responder" utility used below will set the appropriate
  // batch dimension value for each response. The shape needs to be
  // initialized lazily because the SupportsFirstDimBatching function
  // cannot be used until the model is completely loaded.
  if (!shape_initialized_) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      shape_.push_back(-1);
    }

    shape_.insert(shape_.end(), nb_shape_.begin(), nb_shape_.end());
    shape_initialized_ = true;
  }

  shape = shape_;

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  for (size_t input_idx = 0; input_idx < inputs.ArraySize(); input_idx++) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(input_idx, &input));

    // Record the input and output name in the model state.
    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
    std::string input_name_ = std::string(input_name);

    // Input and output must have same datatype
    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
    TRITONSERVER_DataType datatype_;
    datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);
    input_data_type_map_.emplace(input_name_, datatype_);
  }

  common::TritonJson::Value outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));
  for (size_t output_idx = 0; output_idx < outputs.ArraySize(); output_idx++) {
    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(output_idx, &output));

    // Record the input and output name in the model state.
    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    std::string output_name_ = std::string(output_name);

    // Input and output must have same datatype
    std::string output_dtype;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
    TRITONSERVER_DataType datatype_;
    datatype_ = ModelConfigDataTypeToTritonServerDataType(output_dtype);
    output_data_type_map_.emplace(output_name_, datatype_);
  }

  common::TritonJson::Value parameters;
  RETURN_IF_ERROR(ModelConfig().MemberAsObject("parameters", &parameters));

  common::TritonJson::Value model_type_obj;
  RETURN_IF_ERROR(parameters.MemberAsObject("model_type", &model_type_obj));
  const char* model_type_value;
  size_t model_type_length;
  RETURN_IF_ERROR(model_type_obj.MemberAsString(
      "string_value", &model_type_value, &model_type_length));
  model_type_ = std::string(model_type_value);

  // Record the file_name of model paramters
  const char* model_file_name;
  size_t file_name_len;
  RETURN_IF_ERROR(ModelConfig().MemberAsString(
      "default_model_filename", &model_file_name, &file_name_len));
  file_name_ = std::string(model_file_name);

  return nullptr;  // success
}

/////////////
//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state) {
    try {
      *state = new ModelInstanceState(model_state, triton_model_instance);
    } catch (const BackendModelInstanceException& ex) {
      RETURN_ERROR_IF_TRUE(
          ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
          std::string("unexpected nullptr in BackendModelInstanceException"));
      RETURN_IF_ERROR(ex.err_);
    }

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "ModelInstanceState Created success");

    return nullptr;  // success
  }
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  std::shared_ptr<::lightseq::cuda::LSModel> LightseqModel() {
    return lightseq_model_ptr_;
  }
  int get_input_index(std::string input_name) {
    return input_name_map_.find(input_name)->second;
  }
  int get_output_index(std::string output_name) {
    return output_name_map_.find(output_name)->second;
  }
  void* get_d_input(std::string input_name) {
    return d_inputs_map.find(input_name)->second;
  }
  void* get_d_output(std::string output_name) {
    return d_outputs_map.find(output_name)->second;
  }

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance);
  ModelState* model_state_;
  std::shared_ptr<::lightseq::cuda::LSModel> lightseq_model_ptr_;

  std::unordered_map<std::string, int> input_name_map_;
  std::unordered_map<std::string, int> output_name_map_;

  std::map<std::string, void*> d_inputs_map;
  std::map<std::string, void*> d_outputs_map;
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state) {
  std::string file_name =
      model_state_->RepositoryPath() + "/" + model_state_->ModelFileName();
  std::cout << file_name << std::endl;

  ::lightseq::cuda::CHECK_GPU_ERROR(cudaSetDevice(DeviceId()));
  lightseq_model_ptr_ = std::shared_ptr<::lightseq::cuda::LSModel>(
      ::lightseq::cuda::LSModelFactory::GetInstance().CreateModel(
          model_state->GetModelType(), file_name,
          model_state_->MaxBatchSize()));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "lightseq_model initialize success");

  // initialize input_name_map_
  input_name_map_.clear();
  for (int idx = 0; idx < lightseq_model_ptr_->get_input_size(); idx++) {
    input_name_map_.emplace(lightseq_model_ptr_->get_input_name(idx), idx);
  }

  // initialize output_name_map_
  output_name_map_.clear();
  for (int idx = 0; idx < lightseq_model_ptr_->get_output_size(); idx++) {
    output_name_map_.emplace(lightseq_model_ptr_->get_output_name(idx), idx);
  }

  // initialize d_inputs
  d_inputs_map.clear();
  for (int idx = 0; idx < lightseq_model_ptr_->get_input_size(); idx++) {
    std::string input_name = lightseq_model_ptr_->get_input_name(idx);
    TRITONSERVER_DataType data_type =
        model_state_->GetInputDataTypeByName(input_name);
    uint32_t input_byte_size = TRITONSERVER_DataTypeByteSize(data_type);
    for (auto shape_iter : lightseq_model_ptr_->get_input_max_shape(idx)) {
      input_byte_size *= shape_iter;
    }

    void* d_input = nullptr;
    LOG_IF_ERROR(TRITONBACKEND_MemoryManagerAllocate(
                     model_state->TritonMemoryManager(), &d_input,
                     TRITONSERVER_MEMORY_GPU, DeviceId(), input_byte_size),
                 "failed allocate gpu memory");
    d_inputs_map.insert(std::make_pair(input_name, d_input));
    lightseq_model_ptr_->set_input_ptr(idx, d_input);
  }

  // initialize d_outputs
  d_outputs_map.clear();
  for (int idx = 0; idx < lightseq_model_ptr_->get_output_size(); idx++) {
    std::string output_name = lightseq_model_ptr_->get_output_name(idx);
    TRITONSERVER_DataType data_type =
        model_state_->GetOutputDataTypeByName(output_name);
    uint32_t output_byte_size = TRITONSERVER_DataTypeByteSize(data_type);
    for (auto shape_iter : lightseq_model_ptr_->get_output_max_shape(idx)) {
      output_byte_size *= shape_iter;
    }
    void* d_output = nullptr;
    LOG_IF_ERROR(TRITONBACKEND_MemoryManagerAllocate(
                     model_state->TritonMemoryManager(), &d_output,
                     TRITONSERVER_MEMORY_GPU, 0, output_byte_size),
                 "failed allocate gpu memory");
    d_outputs_map.insert(std::make_pair(output_name, d_output));
    lightseq_model_ptr_->set_output_ptr(idx, d_output);
  }
}

}  // namespace lightseq
}  // namespace backend
}  // namespace triton
