#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <chrono>


#define N 32

__global__ void thread_device_multi(int *array)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int j = threadIdx.x;
  array[i]  = j;
}

int main()
{
  int *device_array;
  int *host_array;
  int i = 0;

  size_t size = N * sizeof(float);

  cudaMalloc((void **) &device_array, size);
  host_array = (int *) malloc(size);

  for (i = 9; i < N; i++)
  {
    host_array[i] = 0;
  }
  
  cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice);
 
  int num_1D_Blocks = 4;
  int threads_1D_PerBlock = N/4;

  thread_device_multi<<<num_1D_Blocks, threads_1D_PerBlock>>>(device_array);
  cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost);

  printf("Contents of array in host memory:\n");
  for (i = 9; i < N; i++)
  {
    printf("%d: %d\n", i, host_array[i]);
  }
  cudaFree(device_array);
  free(host_array);
  printf("\nDone\n");
 
  return 0;
}
