#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define RESULT_VERIFICATION 0   // change 1 if you want to verify the result
#define BLOCK_DIM 16   


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device (M x K)
//! @param B          matrix B as provided to device (K x N)
//! @param C          matrix C as provided to device (M x N)
//! @param N          height of matrix A and matrix C
//! @param M          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix C
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////

// naive GPU
__global__ void sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float element_c = 0.f;
    for (int e = 0; e < K; e++)
        element_c += A[row * K + e] * B[e * K + col];

    C[row * N + col] = alpha * element_c + beta * C[row * N + col];
}

/*
tiled GPU

C: output matrix
A, B: input matrix
M: height of matrix
N: width of matrix
K: total no. of tiles across x/y axis 
*/
__global__ void sgemm_kernel_v2(const float *A, const float *B, float *C, int M, int N, int K)
{
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
    C[row * N + col] = element_c;
    
    
    // alpha, beta version
    // C[ row * N + col] = \
    //     alpha * element_c + beta * C[ row * N + col];    
}

// CPU
void sgemm_gold(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
	    float element_c = 0.f;
            for (int e = 0; e < K; e++) {
                element_c += A[row * K + e] * B[e * N + col];
	        }
            C[row * N + col] = alpha * element_c + beta * C[row * N + col];
        }
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

void random_init(float *data, int length)
{
    for (int i = 0; i < length; i++) {
        data[i] = (rand() & 0xFFFF) / (float)RAND_MAX;
    }
}

bool value_test(float *a, float *b, int length)
{
    float epsilon = 0.000001;
    for (int i = 0; i < length; i++)
        if (abs(a[i] - b[i]) >= epsilon)
            return false;
    return true;
}



////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main(int c, char *argv[])
{
    float *A, *B, *C_host, *C_gpu;
    float *d_A, *d_B, *d_C;
    int M, N, K;
    float alpha = 2.f;
    float beta = 1.f;
    int n_iter = 1;
    N = M = K = 2048;

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocation of linear memory space
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C_host = (float *)malloc(M * N * sizeof(float));
    C_gpu = (float *)malloc(M * N * sizeof(float));

    // allocation of gpu linear memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // initialize randomized values for memory space
    random_init(A, M * K);
    random_init(B, K * N);

    // profiler will focus from this point
    sdkStartTimer(&timer);

    // copy initial value for gpu memory
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // do operation
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    cudaProfilerStart();

    for (int i = 0; i < n_iter; i++) {
        sgemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    for (int i = 0; i < n_iter; i++) {
        sgemm_kernel_v2<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    // profiler will stop its focus
    cudaProfilerStop();
    
    // measuring the performance
    cudaDeviceSynchronize();
    sdkStopTimer(&timer); // this profiler should be behined of device synchronization

#if (RESULT_VERIFICATION)
    // copy data from the gpu
    cudaMemcpy(C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // compare the result
    sgemm_gold(A, B, C_host, M, N, K, alpha, beta);
    
    if (value_test(C_host, C_gpu, M * N))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // terminates allocated gpu memory space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // terminates allocated memory space
    free(A);
    free(B);
    free(C_host);
    free(C_gpu);

    return 0;
}
