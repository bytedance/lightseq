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

namespace lightseq {

enum NodeType { VariableNode, IONode, ParametersNode, OperatorNode };

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

extern thread_local ContextPtr thread_context_ptr;

}  // namespace lightseq
