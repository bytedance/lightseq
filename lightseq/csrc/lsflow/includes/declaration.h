#pragma once
#include "memory"
#include "thread"
#include <stdio.h>
#include <fstream>
#include "unordered_set"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <type_traits>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#include "cuda_util.h"
#include "cublas_wrappers.h"

namespace lightseq {

enum class NodeType { Variable, Operator };
// const std::string NodeTypeString[] = {"Variable", "Operator"};
enum VariableType { FixedVariable, SharedVariable, DescendantsVariable };
const std::string VariableTypeString[] = {"FixedVariable", "SharedVariable",
                                          "DescendantsVariable"};

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
