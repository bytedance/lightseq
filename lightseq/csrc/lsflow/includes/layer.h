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

 public:
  Layer(std::string name);
  virtual ~Layer();
  std::string name() { return _name; }

  virtual void forward();
  virtual void backward();

  void set_inputs(std::vector<Variable*> inps) { _inp_var_vec = inps; }
  void set_outputs(std::vector<Variable*> outs) { _out_var_vec = outs; }

  Variable* input(int idx) { return _inp_var_vec[idx]; }
  Variable* output(int idx) { return _out_var_vec[idx]; }

  void clear_fw_flag();

  void clear_bw_flag();

  void tag_fw_flag();

  void tag_bw_flag();

  void gather_root_leaf_var();

  std::vector<Operator*> _op_vec;

  bool macro_inputs_check = false;
  bool macro_outputs_check = false;
};

#define LAYER_PRE_INPUTS(...)                                        \
  set_inputs({__VA_ARGS__}), _context_ptr->enter_layer(this, false), \
      macro_inputs_check = true
      
#define LAYER_POST_OUTPUTS(...)                           \
  set_outputs({__VA_ARGS__}), _context_ptr->exit_layer(), \
      macro_outputs_check = true

}  // namespace lightseq
