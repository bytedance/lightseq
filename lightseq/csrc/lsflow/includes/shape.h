#pragma once
#include "declaration.h"

namespace lightseq {

// This class records the shape information of the tensor and encapsulates some
// methods that may be commonly used.
class Shape {
 private:
  std::vector<int> _shape_vec;
  int _element_size;
  bool _is_calculated;

 public:
  // Default constructor, not part of expected usage.
  Shape() : _shape_vec({-1}), _element_size(0), _is_calculated(false) {}
  Shape(std::vector<int> shape)
      : _shape_vec(shape), _element_size(0), _is_calculated(false) {}
  Shape(const Shape& lx) = default;
  virtual ~Shape() = default;
  const std::vector<int>& view() const { return _shape_vec; }

  // Returns the product of each dimension of shape.
  int element_size();

  // Print shape information.
  void print_shape();
};

}  // namespace lightseq
