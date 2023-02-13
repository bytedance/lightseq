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

class Layer {
 protected:
  ContextPtr _context_ptr;
  std::string _name = "";
  std::vector<Variable*> _root_var_vec = {};
  std::vector<Variable*> _leaf_var_vec = {};

  std::vector<Variable*> _inp_var_vec = {};
  std::vector<Variable*> _out_var_vec = {};

  bool _defined_forward_process = true;
  bool _defined_backward_process = true;

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

  void clear_fw_flag();

  void clear_bw_flag();

  void tag_fw_flag();

  void tag_bw_flag();

  void gather_root_leaf_var();

  std::vector<Operator*> _op_vec;

  bool macro_inputs_check = false;
  bool macro_outputs_check = false;
};

}  // namespace lightseq
