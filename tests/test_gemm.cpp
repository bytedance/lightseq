#include <iostream>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>

int8_t float2int8(float f, float scale) {
    int8_t i = int8_t(f * scale + 0.5);
    if (i < -127) i = -127;
    if (i > 127) i = 127;
    return i;
}

template <typename T, typename S>
float cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                     int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                     S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else if (std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    } else {
        printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    if (status == CUBLAS_STATUS_SUCCESS)
        return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    else
        return -1;
}

void test_int8(int m, int n, int k, int algo, int iteration) {
    int8_t *A, *B;
    int32_t *C;
    cudaMallocManaged(&A, m * k * sizeof(int8_t));
    cudaMallocManaged(&B, k * n * sizeof(int8_t));
    cudaMallocManaged(&C, m * n * sizeof(int32_t));
    for (int i = 0; i < m * k; ++i) A[i] = i % 2;
    for (int i = 0; i < k * n; ++i) B[i] = i % 2;
    int32_t alpha = 1, beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        float time = cublas_gemm_ex(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    B,
                                    A,
                                    C,
                                    n,
                                    k,
                                    n,
                                    &alpha,
                                    &beta,
                                    static_cast<cublasGemmAlgo_t>(algo));
        if (i > 0) total_time += time;
    }
    if (total_time > 0)
        printf("int8 (algo %d): %.3f ms\n", algo, total_time / (iteration - 1));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void test_fp32(int m, int n, int k, int algo, int iteration) {
    float *A, *B, *C;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    for (int i = 0; i < m * k; ++i) A[i] = i % 2;
    for (int i = 0; i < k * n; ++i) B[i] = i % 2;
    float alpha = 1, beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        float time = cublas_gemm_ex(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    B,
                                    A,
                                    C,
                                    n,
                                    k,
                                    n,
                                    &alpha,
                                    &beta,
                                    static_cast<cublasGemmAlgo_t>(algo));
        if (i > 0) total_time += time;
    }
    if (total_time > 0)
        printf("fp32 (algo %d): %.3f ms\n", algo, total_time / (iteration - 1));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void test_fp16(int m, int n, int k, int algo, int iteration) {
    __half *A, *B, *C;
    cudaMallocManaged(&A, m * k * sizeof(__half));
    cudaMallocManaged(&B, k * n * sizeof(__half));
    cudaMallocManaged(&C, m * n * sizeof(__half));
    for (int i = 0; i < m * k; ++i) A[i] = __float2half_rn(i % 2);
    for (int i = 0; i < k * n; ++i) B[i] = __float2half_rn(i % 2);
    __half alpha = __float2half_rn(1.0), beta = __float2half_rn(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        float time = cublas_gemm_ex(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    B,
                                    A,
                                    C,
                                    n,
                                    k,
                                    n,
                                    &alpha,
                                    &beta,
                                    static_cast<cublasGemmAlgo_t>(algo));
        if (i > 0) total_time += time;
    }
    if (total_time > 0)
        printf("fp16 (algo %d): %.3f ms\n", algo, total_time / (iteration - 1));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    int m = 4096, n = 8192, k = 1024;
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int iteration = 10;

    for (int algo = start_algo; algo <= end_algo; ++algo) {
        test_int8(m, n, k, algo, iteration);
    }
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo) {
        test_int8(m, n, k, algo, iteration);
    }

    for (int algo = start_algo; algo <= end_algo; ++algo) {
        test_fp32(m, n, k, algo, iteration);
    }
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo) {
        test_fp32(m, n, k, algo, iteration);
    }

    for (int algo = start_algo; algo <= end_algo; ++algo) {
        test_fp16(m, n, k, algo, iteration);
    }
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo) {
        test_fp16(m, n, k, algo, iteration);
    }
    return 0;
}