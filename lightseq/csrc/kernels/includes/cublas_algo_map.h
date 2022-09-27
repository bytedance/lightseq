/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA FasterTransformer
   This file is adapted from NVIDIA FasterTransformer
*/
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "kernels.h"
#include "cuda_util.h"

#define STRIDE 32
#define BORDER 512
#define IGEMM_T4_CONFIG "/tmp/igemm_configs/igemm_T4.cfg"
#define IGEMM_A100_CONFIG "/tmp/igemm_configs/igemm_A100.cfg"
#define IGEMM_A30_CONFIG "/tmp/igemm_configs/igemm_A30.cfg"
#define IGEMM_A10_CONFIG "/tmp/igemm_configs/igemm_A10.cfg"

typedef struct {
  int algoId, customOption, tile, splitK_val, swizzle, reductionScheme,
      workspaceSize, stages;
  float fp16_time, int8_time, speedup;
  std::string dataOrder;
} cublasLtMatmulAlgo_info;

class cublasAlgoMap {
 private:
  std::map<std::vector<int>, std::map<std::string, cublasLtMatmulAlgo_info> >
      _algo_map;
  std::string _config_filename;

 public:
  explicit cublasAlgoMap(const std::string filename);
  cublasAlgoMap();
  cublasAlgoMap(const cublasAlgoMap& map);
  ~cublasAlgoMap();

  void loadGemmConfig();
  bool isExist(int m, int n, int k);
  cublasLtMatmulAlgo_info findBestAlgo(
      std::map<std::string, cublasLtMatmulAlgo_info> mp);
  cublasLtMatmulAlgo_info getAlgo(int m, int n, int k);
};
