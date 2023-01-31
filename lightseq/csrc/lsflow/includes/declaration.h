#pragma once

#include "memory"
#include "thread"
#include <stdio.h>
#include <fstream>
#include "unordered_set"
#include <unistd.h>
#include "cmath"
#include <math_constants.h>
#include <type_traits>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

#include "kernel_headers.h"

namespace lightseq {

enum class NodeType { Variable, Operator };
// const std::string NodeTypeString[] = {"Variable", "Operator"};
enum VariableType {
  FixedVariable,
  SharedVariable,
  OffsetVariable,
  RegressiveVariable
};
const std::string VariableTypeString[] = {
    "FixedVariable", "SharedVariable", "OffsetVariable", "RegressiveVariable"};

enum class MATRIX_OP {
  Transpose,
  NonTranspose,
};

enum StatusType { Training, Inference, Evaluation };
const std::string StatusTypeString[] = {"Training", "Inference", "Evaluation"};

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

class Allocator;
using AllocatorPtr = std::shared_ptr<Allocator>;

const int MB_SIZE = 1024 * 1024;

}  // namespace lightseq
