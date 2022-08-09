/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA FasterTransformer
   This file is adapted from NVIDIA FasterTransformer
*/
#pragma once

#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define STRIDE 32
#define BORDER 512
#define IGEMM_SM80_CONFIG "igemm_sm80.cfg"

typedef struct {
  int algoId, customOption, tile, splitK_val, swizzle, reductionScheme,
      workspaceSize, stages;
  std::string dataOrder;
} cublasLtMatmulAlgo_info;

class cublasAlgoMap {
 private:
  std::map<std::vector<int>, cublasLtMatmulAlgo_info> _algo_map;
  std::string _config_filename;

 public:
  explicit cublasAlgoMap(const std::string filename);
  cublasAlgoMap(const cublasAlgoMap& map);
  ~cublasAlgoMap();

  void loadGemmConfig();
  bool isExist(const int m, const int n, const int k);
  cublasLtMatmulAlgo_info getAlgo(const int m, const int n, const int k);
};
