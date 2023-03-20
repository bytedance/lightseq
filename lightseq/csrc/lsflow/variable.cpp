#include "node.h"

namespace lightseq {

Variable::Variable(std::string name, cuda::DataType fw_dtype,
                   cuda::DataType bw_dtype)
    : Node(name, NodeType::Variable),
      _mx_shape_size(0),
      _fw_dtype(fw_dtype),
      _bw_dtype(bw_dtype),
      _variable_type(VariableType::FixedVariable) {
  _value.reset(new Tensor("value", fw_dtype));
  if (_context_ptr->is_training()) _grad.reset(new Tensor("grad", bw_dtype));
}

Variable::Variable(std::string name, size_t mx_shape_size,
                   cuda::DataType fw_dtype, cuda::DataType bw_dtype,
                   VariableType vt)
    : Node(name, NodeType::Variable),
      _mx_shape_size(mx_shape_size),
      _fw_dtype(fw_dtype),
      _bw_dtype(bw_dtype),
      _variable_type(vt) {
  _value.reset(new Tensor("value", _fw_dtype, _mx_shape_size));
  if (_context_ptr->is_training() && bw_dtype != cuda::DataType::kNotSupported)
    _grad.reset(new Tensor("grad", _bw_dtype, _mx_shape_size));
  if (vt == VariableType::SharedVariable) {
    return;
  } else if (vt == VariableType::FixedVariable) {
    malloc_memory(_mx_shape_size);
  } else if (vt == VariableType::RegressiveVariable) {
    return;
  } else {
    printf("Error! var %s useless vt %d\n", _name.c_str(), vt);
    exit(-1);
  }
}

Variable::Variable(std::string name, Variable* parent_variable)
    : Node(name, NodeType::Variable),
      _mx_shape_size(0),
      _fw_dtype(parent_variable->fw_dtype()),
      _bw_dtype(parent_variable->bw_dtype()),
      _parent_variable(parent_variable),
      _variable_type(VariableType::OffsetVariable) {
  _value.reset(new Tensor("value", parent_variable->_value));
  if (_context_ptr->is_training()) {
    _grad.reset(new Tensor("grad", parent_variable->_grad));
  }
  parent_variable->add_descendants(this);
}

void Variable::fixed_memory() {
  if (_variable_type == VariableType::OffsetVariable) {
    printf("OffsetVariable should not execute fixed_memory() func!");
    throw std::runtime_error(
        "OffsetVariable should not execute fixed_memory() func!");
    return;
  }
  if (_children_variable.size() && parents().size() > 0) {
    return;
  }
  if (parents().size() > 0 && children().size() > 0) {
    printf("ERROR! this node is not a IONode!\n");
    throw std::runtime_error("ERROR! this node is not a IONode!\n");
  }
  _value->reset_fixed();
  if (_grad) {
    _grad->reset_fixed();
  }
  return;
}

void Variable::swap_tensor(Variable* var_a, Variable* var_b) {
  Tensor temp = *(var_a->_value.get());
  *(var_a->_value.get()) = *(var_b->_value.get());
  *(var_b->_value.get()) = temp;
  if (var_a->_grad && var_b->_grad) {
    Tensor temp = *(var_a->_grad.get());
    *(var_a->_grad.get()) = *(var_b->_grad.get());
    *(var_b->_grad.get()) = temp;
  }
}

void Variable::set_value(char* value_ptr) {
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_value(const char* value_ptr) {
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_grad(char* grad_ptr) {
  if (_context_ptr->is_training()) {
    _grad->reset_fixed();
    _grad->set_tensor(grad_ptr);
  }
}

void Variable::set_shape(Shape shape) {
  _shape = shape;
  _value->set_shape(shape);
  if (_grad != nullptr) _grad->set_shape(shape);
}

void Variable::malloc_memory(size_t size) {
  size_t value_byte_size = size * dtype_size(_fw_dtype);
  size_t grad_byte_size =
      _context_ptr->is_training() ? size * dtype_size(_bw_dtype) : 0;
#ifdef MEM_DEBUG
  printf(
      "Varaible %s malloc memory, value size: %zu "
      "MB, grad size: %zu MB\n",
      name().c_str(), value_byte_size / MB_SIZE, grad_byte_size / MB_SIZE);
#endif
  _variable_type = VariableType::FixedVariable;

  char* value_ptr = _context_ptr->allocator()->malloc_mem(value_byte_size);

  _value->remove_life_cycle();
  _value->set_tensor(value_ptr);
  if (_context_ptr->is_training() && grad_byte_size) {
    char* grad_ptr = _context_ptr->allocator()->malloc_mem(grad_byte_size);
    _grad->remove_life_cycle();
    _grad->set_tensor(grad_ptr);
  }
}

char* Variable::value(bool is_open_interval) {
  return _value->tensor(is_open_interval);
}

char* Variable::grad(bool is_open_interval) {
  return _grad->tensor(is_open_interval);
}

void Variable::update_regress_idx() {
  if (variable_type() != VariableType::RegressiveVariable) {
    return;
  }
  _value->update_life_idx(_context_ptr->regress_begin_idx());
  _value->update_life_idx(_context_ptr->regress_end_idx());
  if (_context_ptr->is_training() && _grad) {
    _grad->update_life_idx(_context_ptr->regress_begin_idx());
    _grad->update_life_idx(_context_ptr->regress_end_idx());
  }
}

bool Variable::enable_override_grad() {
  if (this->_children.size() == 1) {
    return true;
  } else {
    return false;
  }
}

void Variable::add_descendants(Variable* var) {
  _children_variable.insert(var);
}

void Variable::set_offset(int offset, Shape shape) {
  _shape = shape;
  _value->set_offset(offset, shape);
  if (_grad != nullptr) {
    _grad->set_offset(offset, shape);
  }
}

#ifdef DEBUG_MODE
void Variable::print_var(bool is_fw, int size) {
  if (!_context_ptr->is_built()) {
    return;
  }
  if (is_fw) {
    if (_value == nullptr) {
      printf("%s does not have _value object.\n", _name.c_str());
    } else if (value() == nullptr)
      printf("%s value address is nullptr\n", _name.c_str());
    else {
      try {
        _value->print_tensor(size);
      } catch (...) {
        printf("%s variable print tensor value failed!\n", _name.c_str());
      }
    }
  } else {
    if (grad() == nullptr)
      printf("grad address is nullptr\n");
    else {
      try {
        _grad->print_tensor(size);
      } catch (...) {
        printf("%s variable print tensor grad failed!\n", _name.c_str());
      }
    }
  }

  printf("\n");
}
#endif

}  // namespace lightseq
