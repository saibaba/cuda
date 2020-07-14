// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define ITER_ROW 1024
#define ITER_COL 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void matrix_add_cpu(int *a, int *b, int *c, int m, int n) {

    // Add the vector elements a and b to the vector c
    for (int row = 0; row < m ; row++) {
        for (int col = 0; col < n ; col++) {
            int loc = n*row+col;
            *(c+loc) = *(a+loc) + *(b+loc);
        }
    }
}

__global__ void matrix_add_gpu(int *gpu_a, int *gpu_b, int *gpu_c) {
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int loc = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    *(gpu_c+loc) = (*(gpu_a+loc)) + (*(gpu_b+loc));
}

void print_matrix(int *m, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int loc = (cols*row)+col;
            int val = *(m+loc);
            printf("\t%d", val);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {

    int rows = ITER_ROW;
    int cols = ITER_COL;

    int *a, *b, *c;
    int *gpu_a, *gpu_b, *gpu_c;
    
    size_t mem_size = rows * cols * sizeof(int);

    a = (int *)malloc(mem_size);
    b = (int *)malloc(mem_size);
    c = (int *)malloc(mem_size);

    // We need variables accessible to the GPU,
    // so cudaMallocManaged provides these
    gpuErrchk(cudaMalloc(&gpu_a, mem_size));
    gpuErrchk(cudaMalloc(&gpu_b, mem_size));
    gpuErrchk(cudaMalloc(&gpu_c, mem_size));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int loc = cols*row+col;
            *(a+loc) = row+col;
            *(b+loc) = row+col;
        }
    }

    gpuErrchk(cudaMemcpy(gpu_a, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_b, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_c, c, mem_size, cudaMemcpyHostToDevice));

    // Call the CPU function and time it
    auto cpu_start = Clock::now();
    matrix_add_cpu(a, b, c, rows, cols);
    auto cpu_end = Clock::now();
    std::cout << "vector_add_cpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
    << " nanoseconds.\n";
    // print_matrix(c, rows, cols);
    printf("CPU (0, 0): %d\n", *c);
    printf("CPU (3, 17): %d\n", *(c+(13*cols)+17));
    printf("CPU last: %d\n", *(c+(rows*cols)-1));


    *(c+(rows*cols)-1) = 0;
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        int loc = cols*row+col;
        *(c+loc) = 0;
      }
    }

    // Call the GPU function and time it
    // The triple angle brakets is a CUDA runtime extension that allows
    // parameters of a CUDA kernel call to be passed.
    // In this example, we are passing one thread block with ITER GPU threads.

    // why, these dimensions: just for fun: overall we wanted 1024x1024 threads and we can have only 1024 threads/block
    dim3 blocksPerGrid(16, 64);
    dim3 threadsPerBlock(32, 32);

    auto gpu_start = Clock::now();
    matrix_add_gpu <<<blocksPerGrid, threadsPerBlock>>> (gpu_a, gpu_b, gpu_c);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk(cudaDeviceSynchronize());
    auto gpu_end = Clock::now();
    std::cout << "vector_add_gpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count()
    << " nanoseconds.\n";

    gpuErrchk(cudaMemcpy(c, gpu_c, mem_size, cudaMemcpyDeviceToHost));

    printf("GPU (0, 0): %d\n", *c);
    printf("GPU (3, 17): %d\n", *(c+(13*cols)+17));
    printf("GPU last: %d\n", *(c+(rows*cols)-1));
    //print_matrix(c, rows, cols);

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
