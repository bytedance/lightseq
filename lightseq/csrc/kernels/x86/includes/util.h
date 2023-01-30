#pragma once

#include <math_constants.h>
#include <type_traits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

namespace lightseq {
namespace x86 {

enum class MATRIX_OP {
  Transpose,
  NonTranspose,
};

}

}  // namespace lightseq
