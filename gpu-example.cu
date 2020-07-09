#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define ITER 65535

// CPU version of the vector add function
void vector_add_cpu(int *a, int *b, int *c, int n) {
    int i;

    // Add the vector elements a and b to the vector c
    for (i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
    }
}

// GPU version of the vector add function via "__global__" prefix.
// These kind of functions are called kernels in CUDA. When called, they are executed in parallel by N different threads
// as opposed to only once in a regular c++ function
__global__ void vector_add_gpu(int *gpu_a, int *gpu_b, int *gpu_c) {
    int i = threadIdx.x;
    // No for loop needed because the CUDA runtime
    // will thread this ITER times
    gpu_c[i] = gpu_a[i] + gpu_b[i];
}

int main() {

    int *a, *b, *c;
    int *gpu_a, *gpu_b, *gpu_c;

    a = (int *)malloc(ITER * sizeof(int));
    b = (int *)malloc(ITER * sizeof(int));
    c = (int *)malloc(ITER * sizeof(int));

    // We need variables accessible to the GPU,
    // so cudaMallocManaged provides these
    cudaMallocManaged(&gpu_a, ITER * sizeof(int));
    cudaMallocManaged(&gpu_b, ITER * sizeof(int));
    cudaMallocManaged(&gpu_c, ITER * sizeof(int));

    for (int i = 0; i < ITER; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    // Call the CPU function and time it
    auto cpu_start = Clock::now();
    vector_add_cpu(a, b, c, ITER);
    auto cpu_end = Clock::now();
    std::cout << "vector_add_cpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
    << " nanoseconds.\n";

    // Call the GPU function and time it
    // The triple angle brakets is a CUDA runtime extension that allows
    // parameters of a CUDA kernel call to be passed.
    // In this example, we are passing one thread block with ITER GPU threads.
    auto gpu_start = Clock::now();
    vector_add_gpu <<<1, ITER>>> (gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    std::cout << "vector_add_gpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count()
    << " nanoseconds.\n";

    // Free the GPU-function based memory allocations
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    // Free the CPU-function based memory allocations
    free(a);
    free(b);
    free(c);

    return 0;
}
