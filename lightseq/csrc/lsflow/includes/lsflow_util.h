/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/

#pragma once
#include "declaration.h"

namespace lightseq {

/* Print run time, for debug */
void print_time_duration(
    const std::chrono::high_resolution_clock::time_point &start,
    std::string duration_name);

}  // namespace lightseq
