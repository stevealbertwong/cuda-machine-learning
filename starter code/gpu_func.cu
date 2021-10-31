/*

CUDA optimization: 
1. GPU version of feedforward(), backprop()
2. GPU version of mat mul, sigmoid, softmax 

https://github.com/PacktPublishing/Learn-CUDA-Programming/blob/master/Chapter10/10_deep_learning/01_ann/src/layer.cu 

*/
#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K) {
  // TODO
  return 0;
}

/* Helper functions for neural networks */



/*
GPU version of feed forward 
*/

/*

tiled matrix multiplication

nick
https://www.youtube.com/watch?v=3xfyiWhtvZw&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=4&ab_channel=CoffeeBeforeArch 
https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/02_matrix_mul/tiled/mmul.cu 
https://github.com/CoffeeBeforeArch/from_scratch/blob/master/matrixMul/matrix_mul.cu 

learn cuda
https://github.com/PacktPublishing/Learn-CUDA-Programming/blob/master/Chapter07/07_parallel_programming_pattern/01_sgemm_optimization/sgemm.cu 

learn CUDA version

C: output matrix
A, B: input matrix
M: height of matrix
N: width of matrix
K: total no. of tiles across x/y axis 
*/
__global__
void gpu_ff_a1(){
    /*
    GPU blocks, threads within a grid diagram
    */
    int bid_x = blockIdx.x * blockDim.x; // block's x axis within 1 grid
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x; // thd's x axis within 1 block
    int tid_y = threadIdx.y;
    int row = bid_y + tid_y; // global row index == row this thread is in
    int col = bid_x + tid_x;

    float element_c = 0.f;
    
    // 1 tile == 1 shared memory size == 1 block of threads
    // all threads fully utilized within the block
    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    /*    
    
    each for loop:    
    - copy 1 tile of data from global(a, b) to shared memory/tile (s_a, s_b)
        - then each thread uses diff combination of orange lines to compute its element in output matrix
        - only 1 tile share of orange lines + 1 tile share of element is computed each loop
    - matrix A (left to right), B (up to down), 1 tile at 1 time
        - thread of s_tile_A element traverse col of matrix A 1 tile at 1 time (left to right)
        - thread of s_tile_B element traverse row of matrix B 1 tile at 1 time (up to down)
    - carve out the exact tile from global memory matrix A, B to block private s_tile_A, s_tile_B
        - carve out 1 tile from A, 1 tile from B, transfer to s_tiles
        - s_tiles == block private == each block has its own s_tile
            - like each thread has its tid == wont overwrite each other
        - ordered by warp (32 threads at 1 time), i.e. each tile is transfer warp by warp 
        - inter-warps are not in order, BUT sync in order by __syncthreads()

    each thread is responsible for 1 orange element (element_c) in output matrix 
    - each thread will loop thru A's row n B's col orange line, tile by tile, element by element 
    - element_c == thread private == wont overwrite by any other thread 

    k == tile, K == num_tiles, tile_size == BLOCK_DIM 
    */

    // looping over 1 tile at 1 time across matrix, A left to right, B up to down 
    for (int k = 0; k < K; k += BLOCK_DIM) 
    {
        /*
        PART I:

        focus on s_tile_A, s_tile_B to visualize what each block of threads are doing !!!!!
        s_tile_A == block private

        each block of threads carve out the exact tile from global memory matrix A, B to block private s_tile_A, s_tile_B        
        each thread reponsible for 1 element in 1 tile, it goes to the exact element in global matrix

        weird looking since matrix is represented as vector 
        */
        s_tile_A[tid_y][tid_x] = A[ (row * K) + k + tid_x) ]; // global row index + tile_id in this loop + tid
        s_tile_B[tid_y][tid_x] = B[ (k*BLOCK_DIM + tid_y) * N + col ]; 
        
        // 1 tile from A, 1 tile from B, done transfered to s_tiles warp by warp 
        __syncthreads();


        /*
        PART II:

        focus on element_c to visualize what each thread is doing !!!!!
        element_c == thread private

        each thread within 1 block/tile, each compute its output matrix's element
        each thread == diff combination of orange line
        */
        for (int e = 0; e < BLOCK_DIM; e++) // looping over elements wihtin 1 tile
            element_c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x]; 

	  // wait for all threads to finish using current tiles before loading in new ones    
	  __syncthreads();
    }

    // write result from GPU back to CPU
    // element_c == thread private == designated 1 pos in output matrix from the get go
    C[row * N + col] = (1.0 + std::exp(-element_c));
      
}





/*

https://www.youtube.com/watch?v=dJNiHydmVjY&list=PLAtMgFDMfGy2mysjPHN_d1cf9sR1muRkq&index=9&ab_channel=EricDarve 
https://github.com/PacktPublishing/Learn-CUDA-Programming/blob/master/Chapter02/02_memory_overview/04_matrix_transpose/conflict_solved.cu

stanny 213 lecture 9 code

stanny CME 213 version
*/
__global__
void gpu_transpose(int* array_in, int* array_out, size_t n_rows, size_t n_cols) {
  const int warp_id  = threadIdx.y;
  const int lane     = threadIdx.x; // id of thread inside the warp 

  // variable allocated in shared memory
  // +1 == wont hit same bank n result in bank conflict 
  __shared__ int block[warp_size][warp_size+1]; 

  const int bc = blockIdx.x;
  const int br = blockIdx.y;

  // 1. load 32x32 block into shared memory from global
  size_t gc = bc * warp_size + lane; // Global column index
  size_t gr;
  // warp reads 1 line (128 bytes) each loop 
  for(int i = 0; i < warp_size / num_warps; ++i) {
    gr = br * warp_size + i * num_warps + warp_id; // Global row index
    block[i * num_warps + warp_id][lane] = array_in[gr * n_cols + gc];
  }
  // warps that do transpose() can be scheduled to run before load() 
  __syncthreads();

  // 2. transpose routine, but in shared memory
  // shared memory == no cache miss, just care abt not reading same bank 
  // Now we switch to each warp outputting a row, which will read
  // from a column in the shared memory. This way everything remains
  // coalesced.
  gr = br * warp_size + lane;

  for(int i = 0; i < warp_size / num_warps; ++i) {
    gc = bc * warp_size + i * num_warps + warp_id;
    array_out[gc * n_rows + gr] = block[lane][i * num_warps + warp_id];
  }
}





////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////










/*
GPU version of backprop

*/

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


/*
GPU version:
diff = 1/N * (yc - y)

N == int == y.n_cols
yc == arma::mat == a2
y == arma::mat 


GPU accelerated:
element wise matrix multiplication 
element wise matrix diff 

*/
int gpu_bp_diff(const struct cache& bpcache, arma::mat y){

}


__global__
void gpu_bp_diff_kernel(){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
GPU version:
dW2 == diff * bpcache.a[0].t() + reg * nn.W[1];

GPU accelerated:
element wise matrix multiplication
matrix transpose
matrix multiplication  



*/
void gpu_bp_dW2(){



  
}




////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
dB2 == arma::sum(diff, 1);

GPU accelerated:
element wise matrix sum reduction into a vector 

https://www.youtube.com/watch?v=zSWwj5tG5CY&ab_channel=EricDarve stanny tutorial version
run this version of row reduction (kernel 2) num_rows times since on a matrix

squash a matrix into a row 
with each thread summing 1 row left to right

*/
__global__
void gpu_bp_db2_kernel(double *input_matrix, double *output_vector, int num_rows, int num_cols){

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // for loop each row
  for (uint i = 0; i < num_rows, i ++){
    // summing each row 
    for (uint s = num_cols/2; s > 0; s >>= 1){ // s == step
      if(tid < s){
        // i * num_cols == index with row we are on since matrix is represented as array
        output_vector[tid + i * num_cols] += input_matrix[tid + i * num_cols + s];
      }
      // wait for all threads to finish summing 1 row
      __syncthreads();    
    }
  }
}

void gpu_bp_db2(double *input_matrix, double *output_vector, int num_rows, int num_cols)){
  
  dim3 num_threads(num_rows, 1);
  dim3 num_blocks(??, ??);
  gpu_bp_db2_kernel<<< num_blocks, num_threads >>>(input_matrix, output_vector, num_rows, num_cols);
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


/*
da1 = nn.W[1].t() * diff;

GPU accelerated:
matrix multiplication  
*/
gpu_bp_da1()


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);


GPU accelerated:
matrix mod
element wise matrix mod

*/
void gpu_bp_dz1(){


}



////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////





/*
dW1 = dz1 * bpcache.X.t() + reg * nn.W[0];


*/
gpu_bp_dW1()


/*

db1 = arma::sum(dz1, 1)

*/
gpu_bp_db1()
