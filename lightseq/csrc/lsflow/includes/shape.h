#pragma once
#include "declaration.h"
#include "initializer_list"

namespace lightseq {

// This class records the shape information of the tensor and encapsulates some
// methods that may be commonly used.
class Shape {
 private:
  std::vector<size_t> _shape_vec;
  size_t _element_size;
  bool _is_calculated;

 public:
  // Default constructor, not part of expected usage.
  Shape() : _shape_vec({0}), _element_size(0), _is_calculated(false) {}
  Shape(std::vector<size_t> shape)
      : _shape_vec(shape), _element_size(0), _is_calculated(false) {}
  Shape(std::initializer_list<size_t> list) : Shape(std::vector<size_t>(list)) {}
  Shape(const Shape& lx) = default;
  virtual ~Shape() = default;
  const std::vector<size_t>& view() const { return _shape_vec; }

  // Returns the product of each dimension of shape.
  size_t element_size();

  // Print shape information.
  void print_shape();
};

}  // namespace lightseq
