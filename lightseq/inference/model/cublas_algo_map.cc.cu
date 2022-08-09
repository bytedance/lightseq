#pragma once

#include "cublas_algo_map.h"

namespace lightseq {
namespace cuda {}  // namespace cuda

cublasAlgoMap::cublasAlgoMap(const std::string filename)
    : _config_filename(filename) {
  loadGemmConfig();
}

cublasAlgoMap::cublasAlgoMap(const cublasAlgoMap& algo_map)
    : _config_filename(algo_map._config_filename),
      _algo_map(algo_map._algo_map),
{}

cublasAlgoMap::~cublasAlgoMap() { _algo_map.clear(); }

void cublasAlgoMap::loadGemmConfig() {
  FILE* fd;
  fd = fopen(_config_filename.c_str(), "r");
  if (fd == NULL) {
    std::cout << "[WARNING] " << _config_filename
              << " is not found; using default GEMM algo" << std::endl;
    return;
  }

  int m, n, k, algoId, customOption, tile, splitK_val, swizzle, reductionScheme,
      workspaceSize, stages, sm;
  float fp16_time, int8_time, speedup;
  char data_order[50];
  char tmp[1024];
  if (!fgets(tmp, 1024, fd)) {
    printf("[ERROR] fgets fail at %s:%d \n", __FILE__, __LINE__);
    exit(-1);
  }
  while (fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %f %f %f %d %s\n", &m, &n,
                &k, &algoId, &tile, &splitK_val, &reductionScheme, &swizzle,
                &customOption, &stages, &workspaceSize, &fp16_time, &int8_time,
                &speedup, &sm, &data_order) != EOF) {
    std::string dataOrder(data_order);
    std::vector<int> mnk = {m, n, k};
    if (_algo_map.find(mnk) == _algo_map.end()) {
      _algo_map[mnk].algoId = algoId;
      _algo_map[mnk].customOption = customOption;
      _algo_map[mnk].tile = tile;
      _algo_map[mnk].splitK_val = splitK_val;
      _algo_map[mnk].swizzle = swizzle;
      _algo_map[mnk].reductionScheme = reductionScheme;
      _algo_map[mnk].workspaceSize = workspaceSize;
      _algo_map[mnk].stages = stages;
      _algo_map[mnk].dataOrder = dataOrder;
    }
  }
  fclose(fd);
}

bool cublasAlgoMap::isExist(const int m, const int n, const int k) {
  std::vector<int> mnk = {m, n, k};
  return _algo_map.find(mnk) != _algo_map.end();
}

cublasLtMatmulAlgo_info cublasAlgoMap::getAlgo(const int m, const int n,
                                               const int k) {
  if (m >= BORDER) m = ((m + STRIDE - 1) / STRIDE) * STRIDE;
  std::vector<int> mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) {
    return _algo_map[mnk];
  } else {
    cublasLtMatmulAlgo_info tmp_algo;
    tmp_algo.algoId = static_cast<int>(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    tmp_algo.customOption = -1;
    tmp_algo.tile = -1;
    tmp_algo.splitK_val = -1;
    tmp_algo.swizzle = -1;
    tmp_algo.reductionScheme = -1;
    tmp_algo.workspaceSize = -1;
    tmp_algo.stages = -1;
    tmp_algo.dataOrder = "CUBLASLT_ORDER_COL";
    return tmp_algo;
  }
}

}  // namespace lightseq
