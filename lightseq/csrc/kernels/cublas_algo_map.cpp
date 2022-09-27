/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA FasterTransformer
   This file is adapted from NVIDIA FasterTransformer
*/
#include "cublas_algo_map.h"

cublasAlgoMap::cublasAlgoMap(const std::string filename)
    : _config_filename(filename) {
  loadGemmConfig();
}

cublasAlgoMap::cublasAlgoMap() {
  std::string gpu_name = getGPUName();
  if (gpu_name == "T4") {
    _config_filename = IGEMM_T4_CONFIG;
  } else if (gpu_name == "A100") {
    _config_filename = IGEMM_A100_CONFIG;
  } else if (gpu_name == "A30") {
    _config_filename = IGEMM_A30_CONFIG;
  } else if (gpu_name == "A10") {
    _config_filename = IGEMM_A10_CONFIG;
  } else {
    _config_filename = "";
  }
  loadGemmConfig();
}

cublasAlgoMap::cublasAlgoMap(const cublasAlgoMap& algo_map)
    : _config_filename(algo_map._config_filename),
      _algo_map(algo_map._algo_map) {}

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
  std::cout << "Load igemm config from " << _config_filename << std::endl;
  while (fscanf(fd, "%d %d %d %s | %d %d %d %d %d %d %d %d | %f %f %f %d\n", &m,
                &n, &k, &data_order, &algoId, &tile, &splitK_val,
                &reductionScheme, &swizzle, &customOption, &stages,
                &workspaceSize, &fp16_time, &int8_time, &speedup, &sm) != EOF) {
    std::string dataOrder(data_order);
    std::vector<int> mnk = {m, n, k};
    _algo_map[mnk][dataOrder].algoId = algoId;
    _algo_map[mnk][dataOrder].customOption = customOption;
    _algo_map[mnk][dataOrder].tile = tile;
    _algo_map[mnk][dataOrder].splitK_val = splitK_val;
    _algo_map[mnk][dataOrder].swizzle = swizzle;
    _algo_map[mnk][dataOrder].reductionScheme = reductionScheme;
    _algo_map[mnk][dataOrder].workspaceSize = workspaceSize;
    _algo_map[mnk][dataOrder].stages = stages;
    _algo_map[mnk][dataOrder].dataOrder = dataOrder;
    _algo_map[mnk][dataOrder].fp16_time = fp16_time;
    _algo_map[mnk][dataOrder].int8_time = int8_time;
    _algo_map[mnk][dataOrder].speedup = speedup;
  }
  fclose(fd);
}

bool cublasAlgoMap::isExist(int m, int n, int k) {
  std::vector<int> mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) return true;

  if (m >= BORDER) m = ((m + STRIDE - 1) / STRIDE) * STRIDE;
  mnk = {m, n, k};
  return _algo_map.find(mnk) != _algo_map.end();
}

cublasLtMatmulAlgo_info cublasAlgoMap::findBestAlgo(
    std::map<std::string, cublasLtMatmulAlgo_info> mp) {
  cublasLtMatmulAlgo_info best_algo = mp.begin()->second;
  for (auto algo : mp) {
    if (algo.second.int8_time < best_algo.int8_time) {
      best_algo = algo.second;
    }
  }
  return best_algo;
}

cublasLtMatmulAlgo_info cublasAlgoMap::getAlgo(int m, int n, int k) {
  std::vector<int> mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) {
    return findBestAlgo(_algo_map[mnk]);
  }

  if (m >= BORDER) m = ((m + STRIDE - 1) / STRIDE) * STRIDE;
  mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) {
    return findBestAlgo(_algo_map[mnk]);
  } else {
    cublasLtMatmulAlgo_info tmp_algo;
    tmp_algo.algoId = -1;
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
