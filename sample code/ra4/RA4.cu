#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

using std::vector;

__device__ __host__ float f(int i) {
  return -1.; // TODO
}

__global__ void kernel(int N, float *out) {
  // TODO
  // Use the function f above
}

int main(int argc, const char **argv) {
  const int num_threads = 512; /* size of thread block */
  const int N = 100000;        /* matrix size */

  float *d_output;
  
  // TODO
  // Allocate memory for d_output
  // Initialize block_dim and num_blocks

  // Uncomment once block_dim and num_blocks are defined
  // printf("Dimension of thread block: %d\n", block_dim.x);
  // printf("Dimension of grid: %d\n", num_blocks.x);

  // TODO
  // Call kernel

  /* Check that the kernel executed as expected. */
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  vector<float> h_output(N, 0.);
  
  // TODO
  // Transfer data from GPU to CPU

  for (int i = 0; i < N; ++i) {
    // This test should pass once the code is implemented correctly.    
    assert(h_output[i] == f(i));
  }  

  printf("All tests have passed; calculation is correct.\n");

  /* Free memory on the device. */
  // TODO

  printf("Program has completed successfully.\n");

  return 0;
}