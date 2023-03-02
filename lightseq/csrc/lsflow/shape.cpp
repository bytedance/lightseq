#include "shape.h"

namespace lightseq {

size_t Shape::element_size() {
  if (_shape_vec.size() == 1 && _shape_vec[0] == 0) {
    printf("this tensor without shape\n");
    return 0;
  }
  if (_is_calculated) {
    return _element_size;
  }
  size_t product = 1;
  for (int iter : _shape_vec) {
    // if (iter <= 0) {
    //   throw std::runtime_error("this tensor with invalid shape");
    //   return 0;
    // }
    product *= iter;
  }
  _is_calculated = true;
  _element_size = product;
  return _element_size;
}

void Shape::print_shape() {
  printf("shape dim: %zu, element size: %d, each dimension: ",
         _shape_vec.size(), element_size());
  for (int i = 0; i < _shape_vec.size(); i++) {
    printf("%zu", _shape_vec[i]);
    if (i == _shape_vec.size() - 1) {
      printf("\n");
    } else {
      printf(", ");
    }
  }
}

}  // namespace lightseq
