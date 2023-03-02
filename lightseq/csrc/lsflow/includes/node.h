/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/

#pragma once
#include "memory"
#include <cstdlib>
#include <iostream>

#include "declaration.h"
#include "tensor.h"
#include "context.h"
#include "manager.h"
#include "lsflow_util.h"
#include "shape.h"

namespace lightseq {

/*
  - Class:  Node
  - Description:
      We abstract the entire calculation process into a network structure,
      which regard the operation unit and IO as nodes in the graph.
      Node class is the base class representation of nodes and is a virtual base
      class.
  - Implementation file: node.cpp
*/
class Node {
 protected:
  Context* _context_ptr;
  std::string _name;
  NodeType _node_type;

  bool _fw_flag = false;
  bool _bw_flag = false;
  bool _bw_first_flag = false;

  std::vector<Node*> _parents{};
  std::vector<Node*> _children{};
  int _fw_node_idx, _bw_node_idx;
  bool _in_regress_scope = false;

 public:
  Node(std::string name, NodeType nt_);
  std::string name() { return _name; }
  virtual ~Node();

  // NodeType is only divided into two types: Variable, Operator
  NodeType node_type() { return _node_type; }

  // Set the parent node of the current node.
  void set_parents(std::vector<Node*> parents);

  // Add a child node to the current node.
  void add_child(Node* child) { _children.push_back(child); }

  // Pure virtual functions need to be implemented in subclasses.
  virtual void forward() = 0;
  virtual void backward() = 0;

  const std::vector<Node*>& parents() { return _parents; }
  const std::vector<Node*>& children() { return _children; }

  void recursive_forward();
  void recursive_backward();

  void clear_fw_flag() { _fw_flag = false; }
  void tag_fw_flag() { _fw_flag = true; }
  void clear_bw_flag() { _bw_flag = false, _bw_first_flag = true; }
  void tag_bw_flag() { _bw_flag = true; }

  bool is_cover();  // true means assign, false means accumulate
};

/*
  - Class:  Variable
  - Description:
    The Variable class is used to encapsulate the input and output tensors in
    the calculation graph. Lightseq considers the automatic backpropagation
    mechanism, two tensors are recorded in a variable object: value and grad.
    Represents the forward-propagated value and the gradient of the
  back-propagated.
  - implementation file: variable.cpp
*/
class Variable : public Node {
 private:
  VariableType _variable_type;
  Variable* _parent_variable = nullptr;
  std::unordered_set<Variable*> _children_variable;

  DataType _fw_dtype;
  DataType _bw_dtype;

  // If mx_shape is constructed by default, then tensor's memory type is
  // FixedMemory or OffsetMemory.
  size_t _mx_shape_size;
  Shape _shape;

 protected:
  TensorPtr _value = nullptr;
  TensorPtr _grad = nullptr;

 public:
  // Applicable to variables using fixed memory, usually as input or output
  // nodes of the entire network.
  Variable(std::string name, DataType fw_dtype,
           DataType bw_dtype = DataType::kNotSupported);

  /*
    Applicable to the situation of self-developed memory.
    Parameters:
      std::string   name
        Indicates the node name, usually named according
        to the op node that produces the var node.
      Shape   shape
        Represents the shape of the tensor recorded by the var node.
      DataType      fw_dtype
        Data type representing the forward pass tensor.
      DataType      bw_dtype
        Data type representing the backward pass tensor.
      VariableType  vt
        FixedVariable   - The memory is allocated by the var node itself.
        SharedVariable  - The memory space is managed uniformly by
    MemoryManager, which is the core of memory sharing management.
        RegressVariable - Only applicable to tensors that need to be passed
    across steps in autoregressive models.
  */
  Variable(
      std::string name, size_t mx_shape_size, DataType fw_dtype,
      DataType bw_dtype = DataType::kNotSupported,
      VariableType vt = VariableType::SharedVariable);  // for Shared memory

  /*
    Applicable when a variable object is a fragment of another variable object.
    For example, the calculation of qkv is a matrix multiplication operation to
    obtain the continuous tensor of qkv, but subsequent operations need to
    obtain q, k, and v respectively, so the fragments in qkv need to be
    intercepted.
  */
  Variable(std::string name, Variable* parent_variable);

  virtual ~Variable() {}

  virtual void forward() {}
  virtual void backward() {}

  const DataType& fw_dtype() const { return _fw_dtype; }
  const DataType& bw_dtype() const { return _bw_dtype; }

  // This method is to switch the current VariableType to FixedMemory.
  // This method will not execute the memory development logic internally,
  // but will only clear the tensor life cycle information originally
  // registered in the MemoryManager.
  void fixed_memory();

  // Exchange the tensor information of two variable objects,
  // which is used when backup exchange is required such as beam search
  static void swap_tensor(Variable* var_a, Variable* var_b);

  // Set the value pointer and shape information for the variable node,
  // usually used for IO type variable nodes.
  void set_value(char* value_ptr);
  void set_value(const char* value_ptr);

  // Set the grad pointer and shape information for the variable node,
  // usually used for IO type variable nodes.
  void set_grad(char* grad_ptr);

  // Just only set shape for variable object.
  void set_shape(Shape shape);

  // Malloc memory space by itself
  void malloc_memory(size_t size);

  VariableType variable_type() { return _variable_type; }
  std::string variable_type_str() { return VariableTypeString[_variable_type]; }

  /*
    value() / grad() means get the value or grad of this node, when
    is_open_interval is true, it doesn't update the lifecycle of the tensor.
    Please refer to tensor() method in Tensor class for internal logic.
  */
  char* value(bool is_open_interval = false);
  char* grad(bool is_open_interval = false);

  template <typename T>
  T* value(bool is_open_interval = false) {
    return (T*)value(is_open_interval);
  }
  template <typename T>
  T* grad(bool is_open_interval = false) {
    return (T*)grad(is_open_interval);
  }

  // For tensors that need to be passed across steps in the autoregressive
  // model, update their lifetime to fit the autoregressive lifetime range.
  void update_regress_idx();

  bool enable_override_grad();

  // Determine whether the current node is the origin node of an OffsetVariable
  // node.
  bool is_ancestor() { return _children_variable.size(); }

  Variable* ancestor() { return _parent_variable; }
  const std::unordered_set<Variable*>& descendants() const {
    return _children_variable;
  }

  // Set the offset value and shape parameter for OffsetVariable.
  void set_offset(int offset, Shape shape);

  void add_descendants(Variable* var);

  // Identifies that the variable is a tensor that needs to be
  // passed across multiple steps in autoregressive.
  void set_regress_var() { _variable_type = VariableType::RegressiveVariable; }

  friend class Node;

#ifdef DEBUG_MODE
  void print_var(bool is_fw = true, int size = 10);
#endif
};

class Operator : public Node {
 protected:
 public:
  Operator(std::string name);
  virtual ~Operator() {}
  void check_override_grad();

  void set_children(std::vector<Node*> children);

  Variable* child(int index) {
    return static_cast<Variable*>(_children[index]);
  }

  Variable* parent(int index) {
    return static_cast<Variable*>(_parents[index]);
  }
};
}  // namespace lightseq
