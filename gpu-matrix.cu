// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define ITER_ROW 1000
#define ITER_COL 2000

void matrix_add_cpu(int *a, int *b, int *c, int m, int n) {

    // Add the vector elements a and b to the vector c
    for (int row = 0; row < m ; row++) {
        for (int col = 0; col < n ; col++) {
            int loc = n*row+col;
            *(c+loc) = *(a+loc) + *(b+loc);
        }
    }
}

__global__ void vector_add_gpu(int *gpu_a, int *gpu_b, int *gpu_c, int n) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int loc = n*row+col;
    *(gpu_c+loc) = *(gpu_a+loc) + *(gpu_b+loc);
}

int main(int argc, char **argv) {

    int rows = ITER_ROW;
    int cols = ITER_COL;

    if (argc > 1) {
        rows = atoi(argv[1]);
    }
    if (argc > 2) {
        cols = atoi(argv[2]);
    }


    int *a, *b, *c;
    int *gpu_a, *gpu_b, *gpu_c;

    a = (int *)malloc(rows * cols * sizeof(int));
    b = (int *)malloc(rows * cols * sizeof(int));
    c = (int *)malloc(rows * cols * sizeof(int));

    // We need variables accessible to the GPU,
    // so cudaMallocManaged provides these
    cudaMallocManaged(&gpu_a, rows * cols * sizeof(int));
    cudaMallocManaged(&gpu_b, rows * cols * sizeof(int));
    cudaMallocManaged(&gpu_c, rows * cols * sizeof(int));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int loc = cols*row+col;
            *(a+loc) = row+col;
            *(b+loc) = row+col;
        }
    }

    // Call the CPU function and time it
    auto cpu_start = Clock::now();
    matrix_add_cpu(a, b, c, rows, cols);
    auto cpu_end = Clock::now();
    std::cout << "vector_add_cpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
    << " nanoseconds.\n";

    // Call the GPU function and time it
    // The triple angle brakets is a CUDA runtime extension that allows
    // parameters of a CUDA kernel call to be passed.
    // In this example, we are passing one thread block with ITER GPU threads.
    auto gpu_start = Clock::now();
    int numBlocks = 1;
    dim3 threadsPerBlock(rows, cols);

    vector_add_gpu <<<numBlocks, threadsPerBlock>>> (gpu_a, gpu_b, gpu_c, cols);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    std::cout << "vector_add_gpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count()
    << " nanoseconds.\n";

    /* for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int loc = cols*row+col;
            printf("%d\t", *(c+loc));
        }
        printf("\n");
    } */

    printf("%d\n", *(c+(rows*cols)-1));
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
