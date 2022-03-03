// copied from https://github.com/dmlc/dgl/pull/2758
#ifndef DGL_ARRAY_CUDA_DGL_CUB_CUH_
#define DGL_ARRAY_CUDA_DGL_CUB_CUH_

#define CUB_NS_PREFIX namespace ls {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::ls::cub
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#undef CUB_NS_QUALIFIER

#endif
