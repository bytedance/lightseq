#pragma once

#include <type_traits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

namespace lightseq {
namespace x86 {}

template <typename T>
void print_vec(const T *outv, std::string outn, int num_output_ele);

}  // namespace lightseq
