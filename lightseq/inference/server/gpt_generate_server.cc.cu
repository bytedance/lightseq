// Copyright (c) 2019, ByteDance CORPORATION. All rights reserved.

#include <cuda.h>
#include <unistd.h>

#include <string>

#include "../model/gpt_encoder.h"
#include "model_config.pb.h"
#include "../proto/gpt_weight.h"
#include "../server/custom.h"
#include "../server/model_config.h"
#include "../server/model_config_cuda.h"
#include "../tools/util.h"

/**
@file
GPT Language Model server based on tensorrt inference server.
*/

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout
#ifdef FP16_MODE
const lightseq::cuda::OperationType OPTYPE =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType OPTYPE =
    lightseq::cuda::OperationType::FP32;
#endif

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace gptgeneration {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kInputOutputShape,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kCudaDevice,
  kCudaMalloc,
  kCudaMemcpy,
  kCudaExecute,
  kCudaStream,
  kCublas,
  kCpuExecute,
  kWeightLoad,
  kModelSize
};

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(const std::string& instance_name, const ModelConfig& config,
          const int gpu_device);
  ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(const uint32_t payload_cnt, CustomPayload* payloads,
              CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  typedef lightseq::cuda::OperationTypeTraits<OPTYPE> _optraits;
  int FreeCudaBuffers();
  int AllocateCudaBuffers(void** pdata, size_t byte_size);

  int GetInputTensorGPU(CustomGetNextInputFn_t input_fn, void* input_context,
                        const char* name, const size_t expected_byte_size,
                        void* input);

  int ExecuteGPU(const uint32_t payload_cnt, CustomPayload* payloads,
                 CustomGetNextInputFn_t input_fn,
                 CustomGetOutputFn_t output_fn);

  // The name of this instance of the backend.
  const std::string instance_name_;

  // The model configuration.
  const ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // The data-type of the input and output tensors. Must be either
  // INT32 or FP32.
  DataType datatype_;
  int datatype_bytesize_;

  // CUDA memory buffers for input and output tensors.
  void* d_input_;
  void* d_buf_;
  void* d_output_;

  // The contexts executing on a GPU, the CUDA stream to use for the
  // execution.
  cudaStream_t stream_;
  cudaStream_t cache_stream_;
  cublasHandle_t hd_;

  lightseq::cuda::GptWeight<OPTYPE> tw_;
  std::shared_ptr<lightseq::cuda::GptEncoder<OPTYPE>> encoder_;
};

Context::Context(const std::string& instance_name,
                 const ModelConfig& model_config, const int gpu_device)
    : instance_name_(instance_name),
      model_config_(model_config),
      gpu_device_(gpu_device),
      datatype_(DataType::TYPE_INVALID),
      d_input_(nullptr),
      d_buf_(nullptr),
      d_output_(nullptr),
      stream_(nullptr),
      cache_stream_(nullptr),
      hd_(nullptr) {}

Context::~Context() {
  FreeCudaBuffers();

  if (hd_ != nullptr) {
    cublasStatus_t cuerr = cublasDestroy(hd_);
    if (cuerr != CUBLAS_STATUS_SUCCESS) {
      LOG_ERROR << "Failed to destroy cublas handle.";
    }
    hd_ = nullptr;
  }

  if (stream_ != nullptr) {
    cudaError_t cuerr = cudaStreamDestroy(stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: "
                << cudaGetErrorString(cuerr);
    }
    stream_ = nullptr;
  }
  if (cache_stream_ != nullptr) {
    cudaError_t cuerr = cudaStreamDestroy(cache_stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: "
                << cudaGetErrorString(cuerr);
    }
    cache_stream_ = nullptr;
  }
}

int Context::FreeCudaBuffers() {
  if (d_input_ != nullptr) {
    cudaError_t cuerr = cudaFree(d_input_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    d_input_ = nullptr;
  }

  if (d_buf_ != nullptr) {
    cudaError_t cuerr = cudaFree(d_buf_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    d_buf_ = nullptr;
  }

  if (d_output_ != nullptr) {
    cudaError_t cuerr = cudaFree(d_output_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
    d_output_ = nullptr;
  }

  return kSuccess;
}

int Context::AllocateCudaBuffers(void** pdata, size_t byte_size) {
  // Allocate GPU memory buffers large enough for each input and
  // output. For performance we allocate once during initialization
  // instead of doing it each time we execute.
  if (*pdata != nullptr) {
    LOG_ERROR << "given pointer own gpu memory before allocate" << std::endl;
    return kCudaMalloc;
  }
  cudaError_t cuerr = cudaMalloc(pdata, byte_size);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "unable to allocate memory in function AllocateCudaBuffers"
              << cudaGetErrorString(cuerr);
    return kCudaMalloc;
  }
  cuerr = cudaStreamSynchronize(stream_);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "Stream synchronize failed after cudaMalloc"
              << cudaGetErrorString(cuerr) << std::endl;
    return kCudaMalloc;
  }
  return kSuccess;
}

int Context::Init() {
  // Very important to set the CUDA device before performing any
  // CUDA API calls. The device is maintained per-CPU-thread, and
  // the same CPU thread will always be used with this instance of
  // the backend, so only need to set the device once.
  LOG_INFO << "Trtis instance init start" << std::endl;
  cudaError_t cuerr = cudaSetDevice(gpu_device_);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "Failed to set CUDA device to " << gpu_device_ << ": "
              << cudaGetErrorString(cuerr);
    return kCudaDevice;
  }

  const int cuda_stream_priority =
      GetCudaStreamPriority(model_config_.optimization().priority());
  cuerr = cudaStreamCreateWithPriority(&stream_, cudaStreamDefault,
                                       cuda_stream_priority);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "Unable to create stream" << cudaGetErrorString(cuerr);
    return kCudaStream;
  }

  cublasStatus_t cublaserr = cublasCreate(&hd_);
  if (cublaserr != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR << "Failed to creat cublas handle";
    return kCublas;
  }
  cublaserr = cublasSetStream(hd_, stream_);
  if (cublaserr != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR << "Failed to set stream for cublas handle";
    return kCublas;
  }

  if (model_config_.input_size() != 1) {
    return kInputOutputShape;
  }

  datatype_ = model_config_.input(0).data_type();
  if (datatype_ != DataType::TYPE_INT32) {
    return kInputOutputDataType;
  }
  datatype_bytesize_ = GetDataTypeByteSize(datatype_);

  if (model_config_.input(0).name() != "inputs_ids") {
    return kInputName;
  }

  if (model_config_.output_size() != 1) {
    return kInputOutputShape;
  }

  if (model_config_.output(0).data_type() != DataType::TYPE_INT32) {
    return kInputOutputDataType;
  }

  if (model_config_.output(0).name() != "outputs_ids") {
    return kOutputName;
  }

  char* mz = getenv("MODEL_ZOO");
  if (mz == NULL) {
    LOG_ERROR << "plz set environment variable MODEL_ZOO !" << std::endl;
    return kWeightLoad;
  }
  std::string model_path = mz;
  model_path += "/" + model_config_.name();
  std::string res = "load model weight from " + model_path + "/gpt.pb\n";
  LOG_INFO << res;
  res = tw_.initializing(model_path + "/gpt.pb");
  if (!res.empty()) {
    LOG_ERROR << res << std::endl;
    return kWeightLoad;
  }

  int max_batch_size = model_config_.max_batch_size();
  int err;
  err = AllocateCudaBuffers(
      &d_input_, max_batch_size * tw_._max_step * datatype_bytesize_);
  if (err != kSuccess) {
    return err;
  }
  err = AllocateCudaBuffers(
      &d_output_, max_batch_size * tw_._max_step * datatype_bytesize_);
  if (err != kSuccess) {
    return err;
  }

  encoder_ = std::make_shared<lightseq::cuda::GptEncoder<OPTYPE>>(
      max_batch_size, reinterpret_cast<int*>(d_input_),
      reinterpret_cast<float*>(d_output_), reinterpret_cast<int*>(d_output_),
      tw_, stream_, cache_stream_, hd_);
  res = encoder_->check();
  if (!res.empty()) {
    LOG_ERROR << res << std::endl;
    return kModelSize;
  }

  long buf_bytesize = encoder_->compute_buffer_bytesize();
  err = AllocateCudaBuffers(&d_buf_, buf_bytesize);
  if (err != kSuccess) {
    return err;
  }
  encoder_->init_buffer(d_buf_);

  // Wait for all init finish.
  cuerr = cudaStreamSynchronize(stream_);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "failed to init GPU for gpt: " << cudaGetErrorString(cuerr)
              << std::endl;
    return kCudaExecute;
  }
  LOG_INFO << "GPT, release-version[" << __DATE__ << " " << __TIME__
           << "], Trtis instance init succeed!" << std::endl;
  return kSuccess;
}

int Context::GetInputTensorGPU(CustomGetNextInputFn_t input_fn,
                               void* input_context, const char* name,
                               const size_t expected_byte_size, void* input) {
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input', which
  // points to CUDA memory.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    if ((total_content_byte_size + content_byte_size) > expected_byte_size) {
      return kInputSize;
    }

    cudaError_t cuerr = cudaMemcpyAsync(
        reinterpret_cast<char*>(input) + total_content_byte_size, content,
        content_byte_size, cudaMemcpyHostToDevice, stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "failed to copy input values to GPU for gpt: "
                << cudaGetErrorString(cuerr) << std::endl;
      LOG_ERROR << "try to copy " << total_content_byte_size + content_byte_size
                << " bytes from input" << std::endl;
      return kCudaMemcpy;
    }

    total_content_byte_size += content_byte_size;
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int Context::ExecuteGPU(const uint32_t payload_cnt, CustomPayload* payloads,
                        CustomGetNextInputFn_t input_fn,
                        CustomGetOutputFn_t output_fn) {
  // Each payload represents a related set of inputs and required
  // outputs. Each payload may have a different batch size. The total
  // batch-size of all payloads will not exceed the max-batch-size
  // specified in the model configuration.
  if (payload_cnt == 0) {
    return kSuccess;
  }

  std::vector<int64_t> shape(
      payloads[0].input_shape_dims[0],
      payloads[0].input_shape_dims[0] + payloads[0].input_shape_dim_cnts[0]);

  int err;
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // For this payload the expected size of the input and output
    // tensors is determined by the batch-size of this payload.
    const uint64_t batch_seq_len = payload.input_shape_dims[0][0];
    if (batch_seq_len > tw_._max_step) {
      LOG_ERROR << "too long seq_len: " << batch_seq_len
                << ", skip this request" << std::endl;
      return kInputSize;
    }
    const uint64_t batchn_element_count = payload.batch_size * batch_seq_len;
    const uint64_t batchn_byte_size = batchn_element_count * datatype_bytesize_;

    // Copy the input tensors into the appropriate CUDA memory buffer.
    err = GetInputTensorGPU(input_fn, payload.input_context, "inputs_ids",
                            batchn_byte_size, d_input_);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    int sample_step =
        encoder_->run_one_sample(payload.batch_size, batch_seq_len);
    // The output shape is [payload-batch-size, shape] if the model
    // configuration supports batching, or just [shape] if the
    // model configuration does not support batching.
    std::vector<int64_t> output_shape = {payload.batch_size, sample_step};
    int64_t output_bytesize =
        output_shape[0] * output_shape[1] * datatype_bytesize_;

    const char* output_name = "outputs_ids";

    void* obuffer;
    if (!output_fn(payload.output_context, output_name, output_shape.size(),
                   &output_shape[0], output_bytesize, &obuffer)) {
      payload.error_code = kOutputBuffer;
      break;
    }

    // If no error but the 'obuffer' is returned as nullptr, then
    // skip writing this output.
    if (obuffer == nullptr) {
      continue;
    }

    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "failed to launch kernel: " << cudaGetErrorString(cuerr)
                << std::endl;
      payload.error_code = kCudaExecute;
      break;
    }

    cuerr = cudaMemcpyAsync(obuffer, d_output_, output_bytesize,
                            cudaMemcpyDeviceToHost, stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "failed to copy output values from GPU for gpt: "
                << cudaGetErrorString(cuerr) << std::endl;
      payload.error_code = kCudaMemcpy;
      break;
    }
  }

  // Wait for all compute and memcpy to complete.
  cudaError_t cuerr = cudaStreamSynchronize(stream_);
  if (cuerr != cudaSuccess) {
    LOG_ERROR << "failed to synchronize GPU for gpt: "
              << cudaGetErrorString(cuerr) << std::endl;
    return kCudaExecute;
  }
  return kSuccess;
}

int Context::Execute(const uint32_t payload_cnt, CustomPayload* payloads,
                     CustomGetNextInputFn_t input_fn,
                     CustomGetOutputFn_t output_fn) {
  if (gpu_device_ == CUSTOM_NO_GPU_DEVICE) {
    return kCpuExecute;
  } else {
    return ExecuteGPU(payload_cnt, payloads, input_fn, output_fn);
  }
}

/////////////

extern "C" {

int CustomInitialize(const CustomInitializeData* data, void** custom_context) {
  // Convert the serialized model config to a ModelConfig object.
  ModelConfig model_config;
  if (!model_config.ParseFromString(std::string(
          data->serialized_model_config, data->serialized_model_config_size))) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(std::string(data->instance_name), model_config,
                                 data->gpu_device_id);
  int err = context->Init();
  if (err != kSuccess) {
    return err;
  }

  *custom_context = static_cast<void*>(context);

  return kSuccess;
}

int CustomFinalize(void* custom_context) {
  if (custom_context != nullptr) {
    Context* context = static_cast<Context*>(custom_context);
    delete context;
  }

  return kSuccess;
}

const char* CustomErrorString(void* custom_context, int errcode) {
  switch (errcode) {
    case kSuccess:
      return "success";
    case kInvalidModelConfig:
      return "invalid model configuration";
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kInputOutputShape:
      return "model must have one input and one output with the same shape";
    case kInputName:
      return "model inputs must be named 'inputs_ids' and 'INPUT1'";
    case kOutputName:
      return "model outputs must be named 'outputs_ids' and 'OUTPUT1'";
    case kInputOutputDataType:
      return "model inputs and outputs must be TYPE_INT32 "
             "data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kCudaDevice:
      return "cudaSetDevice failed";
    case kCudaMalloc:
      return "cudaMalloc failed";
    case kCudaMemcpy:
      return "cudaMemcpy failed";
    case kCudaExecute:
      return "cuda execution failed";
    case kCudaStream:
      return "failed to create CUDA stream";
    case kCublas:
      return "failed to create Cublas handle";
    case kCpuExecute:
      return "cpu execution failed";
    case kWeightLoad:
      return "load gpt weight in .pb failed";
    case kModelSize:
      return "inappropriate gpt model size";
    default:
      break;
  }

  return "unknown error";
}

int CustomExecute(void* custom_context, const uint32_t payload_cnt,
                  CustomPayload* payloads, CustomGetNextInputFn_t input_fn,
                  CustomGetOutputFn_t output_fn) {
  if (custom_context == nullptr) {
    return kUnknown;
  }

  Context* context = static_cast<Context*>(custom_context);
  return context->Execute(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"

}  // namespace gptgeneration
}  // namespace custom
}  // namespace inferenceserver
}  // namespace nvidia
