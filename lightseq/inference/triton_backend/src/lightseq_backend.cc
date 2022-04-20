// Copyright 2022, Bytedance. All rights reserved.

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "triton_model.h"
#include "triton_utils.h"

namespace triton {
namespace backend {
namespace lightseq {

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." +
               std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
               std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                  .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message,
                                                      &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_Initialize success");

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  // Delete the "global" state associated with the backend.
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
                  .c_str());

  delete state;

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_Finalize success");

  return nullptr;  // success
}

}  // extern "C"

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "doing TRITONBACKEND_ModelInitialize");

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInitialize success");

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize success");

  return nullptr;  // success
}

}  // extern "C"

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "doing TRITONBACKEND_ModelInstanceInitialize");
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceModel initial success");

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelState initial success");

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "ModelInstanceState Create success");
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceInitialize success");

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceFinalize success");

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  int device_id;
  TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
  int devicedChoosed;
  cudaGetDevice(&devicedChoosed);
  if (device_id != devicedChoosed) {
    cudaSetDevice(device_id);
  }

  std::shared_ptr<::lightseq::cuda::LSModel> lightseq_model_ptr =
      instance_state->LightseqModel();

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // 'input_buffer' contains the batched input tensor. The backend can
  // implement whatever logic is necessary to produce the output
  // tensor. This backend simply logs the input tensor value and then
  // returns the input tensor value in the output tensor so no actual
  // computation is needed.

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  for (uint32_t idx = 0; idx < request_count; idx++) {
    TRITONBACKEND_Request* request = requests[idx];
    uint32_t input_count;
    TRITONBACKEND_RequestInputCount(request, &input_count);

    for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(TRITONBACKEND_RequestInputByIndex(
                       request, input_idx /* index */, &input),
                   "failed getting request input");

      const int64_t* shape = nullptr;
      const char* input_name = nullptr;
      TRITONSERVER_DataType datatype;
      uint32_t dims_count;
      uint64_t byte_size;
      uint32_t buffer_count;
      if (input != nullptr) {
        LOG_IF_ERROR(TRITONBACKEND_InputProperties(
                         input, &input_name, &datatype, &shape, &dims_count,
                         &byte_size, &buffer_count),
                     "failed getting input properties");
      }

      // malloc GPU memory by triton api;
      void* d_input = instance_state->get_d_input(input_name);
      void* moved_d_input = d_input;

      for (uint32_t buffer_idx = 0; buffer_idx < buffer_count; buffer_idx++) {
        const void* partial_buffer = nullptr;
        uint64_t buffer_byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        LOG_IF_ERROR(TRITONBACKEND_InputBuffer(
                         input, buffer_idx, &partial_buffer, &buffer_byte_size,
                         &memory_type, &memory_type_id),
                     "failed get input buffer");

        ::lightseq::cuda::CHECK_GPU_ERROR(
            cudaMemcpy(moved_d_input, partial_buffer, buffer_byte_size,
                       cudaMemcpyHostToDevice));
        moved_d_input = (void*)(reinterpret_cast<uint64_t>(moved_d_input) +
                                buffer_byte_size);
      }

      // match triton client input with lightseq input by input_name.
      for (int lightseq_input_idx = 0;
           lightseq_input_idx < lightseq_model_ptr->get_input_size();
           lightseq_input_idx++) {
        if (lightseq_model_ptr->get_input_name(lightseq_input_idx) !=
            input_name) {
          continue;
        }
        lightseq_model_ptr->set_input_shape(
            lightseq_input_idx, std::vector<int>(shape, shape + dims_count));
      }
    }

    lightseq_model_ptr->Infer();

    // create response buffer
    TRITONBACKEND_Response* response = responses[idx];
    for (int output_idx = 0; output_idx < lightseq_model_ptr->get_output_size();
         output_idx++) {
      TRITONBACKEND_Output* output = nullptr;
      void* single_output_buffer = nullptr;
      const std::vector<int> lightseq_shape =
          lightseq_model_ptr->get_output_shape(output_idx);
      std::string output_name = lightseq_model_ptr->get_output_name(output_idx);

      const std::vector<int64_t> triton_shape(lightseq_shape.begin(),
                                              lightseq_shape.end());

      TRITONSERVER_DataType triton_datatype_ =
          model_state->GetOutputDataTypeByName(output_name);
      LOG_IF_ERROR(TRITONBACKEND_ResponseOutput(
                       response, &output, output_name.c_str(), triton_datatype_,
                       triton_shape.data(), triton_shape.size()),
                   "failed create an TRITONBACKEND_OutputBuffer");

      uint32_t total_size = 1;
      for (long unsigned int j = 0; j < triton_shape.size(); j++) {
        total_size *= triton_shape[j];
      }
      uint32_t buffer_byte_size =
          total_size * TRITONSERVER_DataTypeByteSize(triton_datatype_);
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
      int64_t output_memory_type_id = 0;
      LOG_IF_ERROR(TRITONBACKEND_OutputBuffer(
                       output, &single_output_buffer, buffer_byte_size,
                       &output_memory_type, &output_memory_type_id),
                   "failed get a buffer to use to hold the tensor data for the "
                   "output.");

      for (int lightseq_output_idx = 0;
           lightseq_output_idx < lightseq_model_ptr->get_output_size();
           lightseq_output_idx++) {
        if (lightseq_model_ptr->get_output_name(lightseq_output_idx) !=
            output_name) {
          continue;
        }

        const void* d_output = static_cast<const void*>(
            lightseq_model_ptr->get_output_ptr(output_idx));

        ::lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(single_output_buffer,
                                                     d_output, buffer_byte_size,
                                                     cudaMemcpyDeviceToHost));
      }
    }
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  bool supports_first_dim_batching;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                       response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                   "failed to send response");
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  size_t total_batch_size = 0;
  if (!supports_first_dim_batching) {
    total_batch_size = request_count;
  } else {
    for (uint32_t r = 0; r < request_count; ++r) {
      auto& request = requests[r];
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
          "failed getting request input");
      if (input != nullptr) {
        const int64_t* shape = nullptr;
        LOG_IF_ERROR(
            TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape,
                                          nullptr, nullptr, nullptr),
            "failed getting input properties");
        if (shape != nullptr) {
          total_batch_size += shape[0];
        }
      }
    }
  }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     instance_state->TritonModelInstance(), request,
                     (responses[r] != nullptr) /* success */, exec_start_ns,
                     compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting request statistics");
#endif  // TRITON_ENABLE_STATS

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

#ifdef TRITON_ENABLE_STATS
  // Report batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern "C"

}  // namespace lightseq
}  // namespace backend
}  // namespace triton
