
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"
#include "model_base.h"

TRITONSERVER_DataType transform_triton_datatype_to_lightseq(
    ::lightseq::DataType data_type_) {
  switch (data_type_) {
    case ::lightseq::DataType::kNotSupported:
      return TRITONSERVER_TYPE_INVALID;
    case ::lightseq::DataType::kFloat32:
      return TRITONSERVER_TYPE_FP32;
    case ::lightseq::DataType::kInt32:
      return TRITONSERVER_TYPE_INT32;
    case ::lightseq::DataType::kInt64:
      return TRITONSERVER_TYPE_INT64;
    case ::lightseq::DataType::kFloat16:
      return TRITONSERVER_TYPE_FP16;
    case ::lightseq::DataType::kInt8:
      return TRITONSERVER_TYPE_INT8;
    case ::lightseq::DataType::kInt16:
      return TRITONSERVER_TYPE_INT16;
    case ::lightseq::DataType::kByte:
      return TRITONSERVER_TYPE_BYTES;
    case ::lightseq::DataType::kUInt8:
      return TRITONSERVER_TYPE_UINT8;
    case ::lightseq::DataType::kUInt16:
      return TRITONSERVER_TYPE_UINT16;
    case ::lightseq::DataType::kUInt32:
      return TRITONSERVER_TYPE_UINT32;
    case ::lightseq::DataType::kUInt64:
      return TRITONSERVER_TYPE_UINT64;
    case ::lightseq::DataType::kFloat64:
      return TRITONSERVER_TYPE_FP64;
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
  return TRITONSERVER_TYPE_INVALID;
}
