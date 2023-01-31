#include "node.h"

namespace lightseq {

Variable::Variable(std::string name)
    : Node(name, NodeType::Variable),
      _value_byte_size(0),
      _grad_byte_size(0),
      _variable_type(VariableType::FixedVariable) {
  _value.reset(new Tensor("value", 0));
  if (_context_ptr->is_training()) _grad.reset(new Tensor("grad", 0));
}

Variable::Variable(std::string name, size_t value_byte_size,
                   size_t grad_byte_size, VariableType vt)
    : Node(name, NodeType::Variable),
      _value_byte_size(value_byte_size),
      _grad_byte_size(grad_byte_size),
      _variable_type(vt) {
  _value.reset(new Tensor("value", _value_byte_size));
  if (_context_ptr->is_training())
    _grad.reset(new Tensor("grad", _grad_byte_size));
  if (vt == VariableType::SharedVariable) {
    return;
  } else if (vt == VariableType::FixedVariable) {
    malloc_memory(_value_byte_size, _grad_byte_size);
  } else if (vt == VariableType::RegressiveVariable) {
    return;
  } else {
    printf("Error! var %s useless vt %d\n", _name.c_str(), vt);
    exit(-1);
  }
}

Variable::Variable(std::string name, const char* para_ptr, char* grad_ptr)
    : Variable(name, (size_t)0, (size_t)0, VariableType::FixedVariable) {
  _value->set_tensor(para_ptr);
  if (_grad) {
    _grad->set_tensor(grad_ptr);
  }
}

Variable::Variable(std::string name, Variable* parent_variable,
                   size_t offset_value, size_t offset_grad)
    : Node(name, NodeType::Variable),
      _is_descendants(true),
      _parent_variable(parent_variable),
      _variable_type(VariableType::OffsetVariable) {
  _value.reset(new Tensor("value", parent_variable->_value, offset_value));
  if (_context_ptr->is_training()) {
    _grad.reset(new Tensor("grad", parent_variable->_grad, offset_grad));
  }
  parent_variable->add_descendants(this);
}

void Variable::fixed_memory() {
  if (_variable_type == VariableType::OffsetVariable) {
    return;
  }
  if (_children_variable.size() && parents().size() > 0) {
    return;
  }
  if (parents().size() > 0 && children().size() > 0) {
    printf("ERROR! this node is not a IONode!\n");
    exit(-1);
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
  remove_ancestor();
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_value(const char* value_ptr) {
  remove_ancestor();
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_grad(char* grad_ptr) {
  remove_ancestor();
  if (_context_ptr->is_training()) {
    _grad->reset_fixed();
    _grad->set_tensor(grad_ptr);
  }
}

void Variable::malloc_memory(size_t value_byte_size, size_t grad_byte_size) {
#ifdef MEM_DEBUG
  printf("Varaible %s malloc memory, value size: %zu MB, grad size: %zu MB\n",
         name().c_str(), value_byte_size / MB_SIZE, grad_byte_size / MB_SIZE);
#endif
  _value_byte_size = value_byte_size;
  _grad_byte_size = grad_byte_size;
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
void Variable::remove_descendants(Variable* var) {
  _children_variable.erase(var);
}

void Variable::set_ancestor(Variable* parent_variable, size_t offset_value,
                            size_t offset_grad) {
  if (_parent_variable != nullptr && _parent_variable != parent_variable) {
    printf("error! var %s with two ancestor!\n", name().c_str());
    printf("new parent_variable: %s\n", parent_variable->_name.c_str());
    printf("original parent_variable: %s\n", _parent_variable->_name.c_str());
    exit(-1);
  } else if (_parent_variable == parent_variable) {
    return;
  }
  _is_descendants = true;
  _parent_variable = parent_variable;
  _variable_type = VariableType::OffsetVariable;
  _value->set_offset(parent_variable->_value, offset_value);
  if (_context_ptr->is_training()) {
    _grad->set_offset(parent_variable->_grad, offset_grad);
  }
  parent_variable->add_descendants(this);
}

void Variable::remove_ancestor() {
  if (_is_descendants) {
    _is_descendants = false;
    _parent_variable->remove_descendants(this);
    _parent_variable = nullptr;
    _value->remove_offset();
    if (_grad) {
      _grad->remove_offset();
    }
  }
}

void Variable::set_offset(size_t offset_value, size_t offset_grad) {
  _value->set_offset(offset_value);
  if (_grad != nullptr) {
    _grad->set_offset(offset_grad);
  }
}

#ifdef DEBUG_MODE
void Variable::print_var(bool is_fw) {
  if (is_fw) {
    if (value() == nullptr)
      printf("value address is nullptr\n");
    else
      _value->print_tensor(10);
  } else {
    if (grad() == nullptr)
      printf("grad address is nullptr\n");
    else
      _grad->print_tensor(10);
  }

  printf("\n");
}
#endif

}  // namespace lightseq
