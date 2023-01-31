#pragma once
#include "declaration.h"

namespace lightseq {

/* Print run time, for debug */
void print_time_duration(
    const std::chrono::high_resolution_clock::time_point &start,
    std::string duration_name);

}  // namespace lightseq
