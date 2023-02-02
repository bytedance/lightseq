/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA FasterTransformer
   This file is adapted from NVIDIA FasterTransformer
*/
#include "cublas_algo_map.h"
namespace lightseq {
namespace cuda {
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

cublasAlgoMap::~cublasAlgoMap() {
  _algo_map.clear();
  CHECK_GPU_ERROR(cudaFree(_workspace));
}

bool cublasAlgoMap::fileExist(std::string path) {
  if (fopen(path.c_str(), "r") == NULL) return false;
  return true;
}

/* Priority:
1. default dir
2. given url
3. default url
4. default algo
*/
void cublasAlgoMap::getGemmConfig() {
  if (fileExist(DEFAULT_DIR + _config_filename)) {
    std::cout << "Get igemm config from " << DEFAULT_DIR + _config_filename
              << std::endl;
    return;
  }

  std::string command = "mkdir -p " + DEFAULT_DIR;
  system(command.c_str());

  const char* config_url_cstr = std::getenv("GEMM_CONFIG_URL");
  std::string config_url = (config_url_cstr == nullptr ? "" : config_url_cstr);
  if (config_url.size() > 0) {
    command = "wget -nc " + config_url + " -P " + DEFAULT_DIR;
    system(command.c_str());
    if (fileExist(DEFAULT_DIR + _config_filename)) {
      std::cout << "Get igemm config from " << config_url << std::endl;
      return;
    }
  }

  command = "wget -nc " + DEFAULT_URL + _config_filename + " -P " + DEFAULT_DIR;
  system(command.c_str());

  if (fileExist(DEFAULT_DIR + _config_filename)) {
    std::cout << "Get igemm config from " << DEFAULT_URL << std::endl;
    return;
  }
}

void cublasAlgoMap::loadGemmConfig() {
  getGemmConfig();

  FILE* fd;
  fd = fopen((DEFAULT_DIR + _config_filename).c_str(), "r");
  if (fd == NULL || _config_filename.size() == 0) {
    std::cout << "[WARNING] " << DEFAULT_DIR + _config_filename
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
  _workspace_size = 0;
  std::cout << "Load igemm config from " << DEFAULT_DIR + _config_filename
            << std::endl;
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
    _workspace_size = std::max(_workspace_size, workspaceSize);
  }
  fclose(fd);
  if (_workspace_size > 0) {
    _workspace_size = sizeof(char*) * _workspace_size;
    CHECK_GPU_ERROR(cudaMalloc((void**)&_workspace, _workspace_size));
  }
}

bool cublasAlgoMap::isExist(int m, int n, int k) {
  std::vector<int> mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) return true;

  if (m >= BORDER) m = ((m + STRIDE - 1) / STRIDE) * STRIDE;
  mnk = {m, n, k};
  return _algo_map.find(mnk) != _algo_map.end();
}

cublasLtMatmulAlgo_info cublasAlgoMap::defaultAlgo() {
  cublasLtMatmulAlgo_info default_algo;
  default_algo.algoId = -1;
  default_algo.customOption = -1;
  default_algo.tile = -1;
  default_algo.splitK_val = -1;
  default_algo.swizzle = -1;
  default_algo.reductionScheme = -1;
  default_algo.workspaceSize = -1;
  default_algo.stages = -1;
  default_algo.dataOrder = "CUBLASLT_ORDER_COL";
  return default_algo;
}

cublasLtMatmulAlgo_info cublasAlgoMap::findBestAlgo(
    std::map<std::string, cublasLtMatmulAlgo_info> mp, std::string data_order) {
  if (data_order.size() > 0) {
    if (mp.find(data_order) != mp.end()) {
      return mp[data_order];
    } else {
      return defaultAlgo();
    }
  } else {
    cublasLtMatmulAlgo_info best_algo = mp.begin()->second;
    for (auto algo : mp) {
      if (algo.second.int8_time < best_algo.int8_time) {
        best_algo = algo.second;
      }
    }
    return best_algo;
  }
}

cublasLtMatmulAlgo_info cublasAlgoMap::getAlgo(int m, int n, int k,
                                               std::string data_order) {
  std::vector<int> mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) {
    return findBestAlgo(_algo_map[mnk], data_order);
  }

  if (m >= BORDER) m = ((m + STRIDE - 1) / STRIDE) * STRIDE;
  mnk = {m, n, k};
  if (_algo_map.find(mnk) != _algo_map.end()) {
    return findBestAlgo(_algo_map[mnk], data_order);
  } else {
    return defaultAlgo();
  }
}

void* cublasAlgoMap::get_workspace() { return _workspace; }

int cublasAlgoMap::get_workspace_size() { return _workspace_size; }
}  // namespace cuda
}  // namespace lightseq
