#pragma once
#define ls_cuda 0
#define ls_x86 1
#define ls_arm 2

#include "memory"
#include "thread"
#include <stdio.h>
#include <fstream>
#include "unordered_set"
#include <unistd.h>

#include "headers.h"

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

const int MB_SIZE = 1024 * 1024;

}  // namespace lightseq
