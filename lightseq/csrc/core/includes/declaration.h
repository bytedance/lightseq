#pragma once
#include "memory"
#include "thread"

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

thread_local ContextPtr thread_context_ptr;

}