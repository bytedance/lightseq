/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/

#pragma once
#include "memory"
#include "thread"
#include <stdio.h>
#include <fstream>
#include "unordered_set"
#include <unistd.h>
#include "cmath"
#include <type_traits>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include "model_base.h"

#include "kernel_headers.h"

namespace lightseq {

#ifdef FP16_MODE
typedef __half OpType_;
#else
typedef float OpType_;
#endif

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

class Shape;

class Allocator;
using AllocatorPtr = std::shared_ptr<Allocator>;

const int MB_SIZE = 1024 * 1024;

#define CHECK_DTYPE(dtype, base_type) (dtype == g_dtype<base_type>())

enum class GenerateMethod { Topk, Topp, BeamSearch, UnDefined };

}  // namespace lightseq
