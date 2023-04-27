#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <type_traits>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#include "kernels.h"
#include "embKernels.h"
#include "gptKernels.h"
#include "transformerKernels.h"
#include "cuda_util.h"
#include "cublas_wrappers.h"
#include "llama_kernels.h"
