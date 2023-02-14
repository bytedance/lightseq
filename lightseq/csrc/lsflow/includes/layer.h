/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/
#pragma once
#include "iostream"
#include "map"

#include "declaration.h"
#include "context.h"
#include "node.h"

namespace lightseq {

/*
  - Class: Layer
  - Description:
      Layer is the encapsulation of some common network structures. Layers can
      be nested and connected to each other, which reduces the development cost
      of network structure. All nodes of the layer are recorded in the Layer
      object (including output nodes, excluding input nodes). When the layer
      object performs forward propagation, it will automatically complete the
      data calculation update for all nodes of the layer.
*/
class Layer {
 protected:
  ContextPtr _context_ptr;
  std::string _name = "";

  std::vector<Variable*> _inp_var_vec = {};
  std::vector<Variable*> _out_var_vec = {};

 public:
  Layer(std::string name);
  virtual ~Layer();
  std::string name() { return _name; }

  virtual void forward() final;
  virtual void backward() final;

  virtual void forward_process() {}
  virtual void backward_process() {}

  void set_inputs(std::vector<Variable*> inps);
  void set_outputs(std::vector<Variable*> outs);

  Variable* input(int idx);
  Variable* output(int idx);

  // Clear the forward propagation mark, you need to ensure that the mark is
  // cleared before each execution of forward.
  void clear_fw_flag();

  // Clear the backward propagation mark, you need to ensure that the mark is
  // cleared before each execution of backward.
  void clear_bw_flag();

  // Mark that the operation in this layer has been completed and does not need
  // to be executed again.
  void tag_fw_flag();
  void tag_bw_flag();

  std::vector<Operator*> _op_vec;

  bool macro_inputs_check = false;
  bool macro_outputs_check = false;
};

}  // namespace lightseq
