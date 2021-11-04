#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../utils/types.h"

struct event_pair {
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair* p) {
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair* p) {
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

int myGEMM(real* A, real* B, real* C, real* alpha, real* beta, int M, int N,
           int K);



/*
Q: std::vector<double*> instread of double* ?
A: seem not much difference in accessing data 

*/
class CPUData {
public:    
    // everything here is already divided into batches

    /* CPU -> GPU */
    // for each ps push to its GPU core when feed_forward()
    double* X; // training samples this ps is assigned 
    double* y;

    /* GPU -> CPU */
    // for each ps to store ouput in CPU from backprop() in GPU
    double* dW1;
    double* dW2;
    double* db1;
    double* db2;

    // MPI_Allreduced() on CPU for gradient descent 
    double* dW1_reduced;
    double* dW2_reduced;
    double* db1_reduced;
    double* db2_reduced;    

    // constructor
    CPUData(int batch_size, int num_features, int num_neurons, int num_classes){
      double* X = (double *) malloc(sizeof(double) * batch_size * num_features);
      double* y = (double *) malloc(sizeof(double) * batch_size * num_classes);
      double* dW1, dW1_reduced = (double *) malloc(sizeof(double) * num_features * num_neurons);
      double* dW2, dW2_reduced = (double *) malloc(sizeof(double) * num_neurons * num_classes);
      double* db1, db1_reduced = (double *) malloc(sizeof(double) * num_neurons);
      double* db2, db2_reduced = (double *) malloc(sizeof(double) * num_classes);
    }

    // destructor 
    ~CPUData(){
      free(X);
      free(y);
      free(dW1);
      free(dW2);
      free(db1);
      free(db2);
      free(dW1_reduced);
      free(dW2_reduced);
      free(db1_reduced);
      free(db2_reduced);
    }


}


class GPUData {
public:
    // everything here is already divided into batches
    // for computing in GPU when feed_forward() + backprop()
    // GPU accelerated all types of vector, matrix operation !!!
    // controlled via CPU
    double* X; // training samples this ps is assigned 
    double* y;
    double* W1;
    double* W2;
    double* b1;
    double* b2;

    double* dW1;
    double* dW2;
    double* db1;
    double* db2;

    // they could be in CPU
    // just GPU global memory is much faster
    // no CPU -> GPU or GPU -> CPU for these guys 
    double* yh;
    double* y_diff;
    double* A1;
    double* A2; 
    double* Z1; 
    double* Z2;
    double* dA1; 
    double* dZ1;

    int batch_size, num_pixels, num_neurons, num_classes;

    // constructor
    GPUData(int batch_size, int num_features, int num_neurons, int num_classes) : batch_size(batch_size), num_features(num_features), num_neurons(num_neurons), num_classes(num_classes) {
      cudaMalloc((void **) &X,    sizeof(double) * batch_size * num_featuresP);
      cudaMalloc((void **) &y,    sizeof(double) * batch_size * num_classes);
      cudaMalloc((void **) &yh,   sizeof(double) * batch_size * num_classes);
      cudaMalloc((void **) &A1,   sizeof(double) * batch_size * num_neurons);
      cudaMalloc((void **) &dA1,  sizeof(double) * batch_size * num_neurons);
      cudaMalloc((void **) &Z1,   sizeof(double) * batch_size * num_neurons);
      cudaMalloc((void **) &dZ1,  sizeof(double) * batch_size * num_neurons);
      cudaMalloc((void **) &A2,   sizeof(double) * batch_size * num_classes);
      cudaMalloc((void **) &Z2,   sizeof(double) * batch_size * num_classes);
      cudaMalloc((void **) &W1,   sizeof(double) * num_neurons * num_features);
      cudaMalloc((void **) &dW1,  sizeof(double) * num_neurons * num_features);
      cudaMalloc((void **) &W2,   sizeof(double) * num_neurons * num_classes);
      cudaMalloc((void **) &dW2,  sizeof(double) * num_neurons * num_classes);
      cudaMalloc((void **) &b1,   sizeof(double) * num_neurons); 
      cudaMalloc((void **) &db1,  sizeof(double) * num_neurons); 
      cudaMalloc((void **) &b2,   sizeof(double) * num_classes);
      cudaMalloc((void **) &db2,  sizeof(double) * num_classes);
      cudaMalloc((void **) &y_diff, sizeof(double) * batch_size * num_classes);
    }

    ~device_cache() {
        cudaFree(X);
        cudaFree(y);
        cudaFree(yh);
        cudaFree(A1);
        cudaFree(dA1);
        cudaFree(Z1);
        cudaFree(dZ1);
        cudaFree(A2);
        cudaFree(Z2);
        cudaFree(W1);
        cudaFree(dW1);
        cudaFree(W2);
        cudaFree(dW2);
        cudaFree(b1);
        cudaFree(db1);
        cudaFree(b2);
        cudaFree(db2);
        cudaFree(y_diff);
    }




}










#endif
