#pragma once
#include "memory"
#include "thread"
#include <stdio.h>
#include <fstream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include "cuda_util.h"
#include "cublas_wrappers.h"

#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include "unordered_set"

namespace lightseq {

enum class NodeType { Variable, Operator };

class Node;

class Operator;

class Variable;

class Layer;
using LayerPtr = std::shared_ptr<Layer>;

class Context;
using ContextPtr = std::shared_ptr<Context>;

class MemoryManager;
using MemoryManagerPtr = std::shared_ptr<MemoryManager>;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace lightseq
