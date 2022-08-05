#include <iostream>
#include <cuda_runtime.h>
using namespace std;

int main() {
  int devID = 0;
  cudaSetDevice(devID);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, devID);
  printf("%d", devProp.major * 10 + devProp.minor);
  return 0;
}
