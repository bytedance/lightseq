
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"
#include "model_base.h"

TRITONSERVER_DataType transform_triton_datatype_to_lightseq(
    ::lightseq::cuda::DataType data_type_) {
  switch (data_type_) {
    case ::lightseq::cuda::DataType::kNotSupported:
      return TRITONSERVER_TYPE_INVALID;
    case ::lightseq::cuda::DataType::kFloat32:
      return TRITONSERVER_TYPE_FP32;
    case ::lightseq::cuda::DataType::kInt32:
      return TRITONSERVER_TYPE_INT32;
    case ::lightseq::cuda::DataType::kInt64:
      return TRITONSERVER_TYPE_INT64;
    case ::lightseq::cuda::DataType::kFloat16:
      return TRITONSERVER_TYPE_FP16;
    case ::lightseq::cuda::DataType::kInt8:
      return TRITONSERVER_TYPE_INT8;
    case ::lightseq::cuda::DataType::kInt16:
      return TRITONSERVER_TYPE_INT16;
    case ::lightseq::cuda::DataType::kByte:
      return TRITONSERVER_TYPE_BYTES;
    case ::lightseq::cuda::DataType::kUInt8:
      return TRITONSERVER_TYPE_UINT8;
    case ::lightseq::cuda::DataType::kUInt16:
      return TRITONSERVER_TYPE_UINT16;
    case ::lightseq::cuda::DataType::kUInt32:
      return TRITONSERVER_TYPE_UINT32;
    case ::lightseq::cuda::DataType::kUInt64:
      return TRITONSERVER_TYPE_UINT64;
    case ::lightseq::cuda::DataType::kFloat64:
      return TRITONSERVER_TYPE_FP64;
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
  return TRITONSERVER_TYPE_INVALID;
}
