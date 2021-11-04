/*
review backprop - cs213n
https://www.youtube.com/watch?v=i94OvYb6noo&t=3868s&ab_channel=AndrejKarpathy 


X == input 
Y == answer
b[i] == row vector biases of the i^th layer
yc == output after feedforward 
z1 == X * W1 
z2 == X * W2
a1 == activation layer 1 == after sigmoid 
a2 == activation layer 2 == after softmax 
regularization == dampen z1 being just X * W1, self introduced error 
dW2 == gradient to update W2 == error introduced by W2

dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);



matrix multiplication optimize:


NOTE: 

for Mac to include <mpi.h>
https://github.com/openai/baselines/issues/114 
sudo apt install libopenmpi-dev 
brew install openmpi 

*/
#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"
#include "mpi.h"
#include "utils/common.h"


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

real norms(NeuralNetwork& nn) {
  real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
  arma::Mat<real> A, B, C, D;

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0) {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////



/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}


/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1); // add gate
  arma::Mat<real> da1 = nn.W[1].t() * diff; // t() == transpose()
  /*
  
  
  */
  arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg) {
  int N = yc.n_cols;
  real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  real data_loss = ce_sum / N;
  real reg_loss = 0.5 * reg * norms(nn);
  real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads) {
  real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h; // +h / 0.00001
        feedforward(nn, X, numcache);
        real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h; // -h / 0.00001
        feedforward(nn, X, numcache);
        real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j) {
      real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        write_cpudata_tofile(nn, iter);
      }

      iter++;
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



/*
need to GPU every step in backprop !!!


1. z1 = W1*x + b1
2. a1 = sigmoid(z1)
3. z2 = W2*a1 + b2 
4. yc = a2 = softmax(z2)


*/
void gpu_feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache){
  
  gpu_ff_a1(); // mat mul + signmoid 
  gpu_ff_z2();
  gpu_ff_yc();

}



/*
need to GPU every step in backprop !!!


1. diff = 1/N * (yc - y)
2. gradient dW2 = diff * a1.T + reg * W2
3. 

*/
void gpu_backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads){

  gpu_bp_diff();
  
  gpu_bp_dW2()
  gpu_bp_db2()
  gpu_bp_da1()
  gpu_bp_dz1()
  gpu_bp_dW1()
  gpu_bp_db1()

}

void gpu_gradientdescent(){

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

MPI tutorial

https://github.com/CoffeeBeforeArch/practical_parallelism_in_cpp/blob/master/parallel_algorithms/gaussian_elimination/mpi/naive/gaussian.cpp 
https://github.com/CoffeeBeforeArch/practical_parallelism_in_cpp/blob/master/parallel_algorithms/gaussian_elimination/mpi/cyclic_striped_mapping/gaussian.cpp 
stanny 213 lecture 16-18 code

https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/ 


identical to MPI_Reduce() withthe exception that it does not need a root process id 
(since the results are distributed to all processes).



*/
/*
 * Train the neural network nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 * 
 * MPI procedures:
 * 1. divide input batches of images n MPI_scatter() 1 batch to each MPI node
 * 2. GPU compute each batch of images' contribution to W's updates
 * 3. allreduce() W's updates, broadcast() to all MPI nodes 
 * 
 * cudamemcpy procedures
 * 1. alloc space for host copies of a, b, c and init values
 * 2. alloc space for device copies of d_a, d_b, d_c
 * 3. cudamemcpy() host copies to device
 * 4. pass device copies as GPU function's input
 * 
 * overall strategy: 
 * 1. setup: 
 *    a. malloc() MPI data + host data + cudaMalloc() device data
 *      - every data in ML pipeline have 2 copies
 *        - 1 on CPU
 *        - 1 on GPU
 * 
 *      - MPI data (input on CPU)
 *        - to store received MPI on CPU
 *        - divided CPU ML data for each GPU core 
 *        - inputs for GPU to compute feed_forward(), backprop() 
 * 
 *      - GPU/device data (input + output on GPU)
 *        - GPU data on GPU, controlled via CPU
 *        - inputs for GPU to compute feed_forward(), backprop() 
 *        - outputs that GPU passes to CPU
 * 
 *      - CPU/host data (output on CPU)
 *        - GPU data on CPU, controlled via CPU
 *        - to store output of GPU backprop()
 * 
 * 2. for each epoch: 
 *    a. MPI between CPUs
 *      - CPU ML data -> divided CPU ML data for each GPU core
 *      - root ps MPI batches of input images n output classes to non root nodes
 *    b. cudaMemcpy() CPU to GPU: 
 *      - divided CPU ML data for each GPU core -> GPU/device data
 *    c. cudaMemcpy() GPU to CPU: 
 *      - GPU/device data -> CPU/host data
 *    d. MPI_Allreduce() at local CPU
 *      - CPU/host data -> sum of CPU/host data (every node has a copy)
 *      - allreduce() W's updates, broadcast() to all MPI nodes 
 *      - i.e. each node sends its CPU to all, while receving all and then sum
 *    e. run() the same gradient descent at every CPU
 *      - MPI_Allreduce() copies the same gradients to all ps
 *      - every ps runs same, so same copy for next epoch
 * 
 * 
 * MPI_Scatterv() vs MPI_Scatter() vs MPI_Bcast()
 * - MPI_Scatterv() == send diff things of diff sizes to diff ps 
 * - MPI_Scatter() == send diff things to diff ps 
 * - MPI_Bcast() == send/receive same thing to diff ps 
 * 
 * NOTE:
 * - both sender n receiver are running the same code
 *    - root also need do the same work as non root in addition of being root
 * - every MPI call is sync
 *    - but could be root as sender, non root as receiver
 *    - or every MPI node as both sender and receiver 
 * 
 * HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function.
 * 
 * 
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check,
                    int print_every, int debug) {
  int rank, num_procs; // set by bash.sh
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs)); // no. of processes
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank)); // get possess_id / rank

  /*
  only root ps knows X's no. of cols ?? 
  root ps broadcast total no. training samples to non root ps
  this call is sync, blocking until all ps received N

  https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/ 

  N: payload to send/receive
  1: size of payload
  0: receiver/sender process id, 0 == root process
  */
  int N = (rank == 0) ? X.n_cols : 0; // N == no. samples 
  int num_features = nn.H[0];
  int num_neurons = nn.H[1];
  int num_classes = nn.H[2];
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  std::ofstream error_file;
  error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;

  /*
  1a. malloc() MPI data + host data + cudaMalloc() device data
  */

  // MPI data (input on CPU) + CPU host data (output on CPU) -> mostly as placeholder
  CPUData CPU_data(batch_size, num_features, num_neurons, num_classes);
  // GPU data (input + output on GPU) -> mostly for control GPU data via CPU
  GPUData GPU_data(batch_size, num_features, num_neurons, num_classes);


  int iter = 0; // debugging, goes from 0 to epochs*num_batches
  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size; // N: total training samples
    for (int batch = 0; batch < num_batches; ++batch) {

      /*
      2a. root ps divides batches of input images to MPI nodes

      this call is sync !!!!!!!
      every process blocked until root ps finished "scattering" to all ps

      TODO: 
      MPI_Scatterv()
      https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node72.html 
      MPI_Scatter() can only send same size data

      1. starting ptr of matrix root ps to scatter to all ps
      2. send_counts, send length, sub matrix each process is assigned (N * no. of rows) 
      3. displacement/stride 

      NOTE:
      1st batch sent to root, then for loop 2nd to 2nd ps, 3rd to 3rd ps etc. 

      */
     
     int batch_index = batch * batch_size; // 4 GPU core can only deal w portion of batch each loop
     int process_share = batch_size/num_procs; // depending no. of process, each ps might compute many batches
     int last_process_share = (N - batch_index)/num_procs; // last ps likely get less share of work
     int final_process_share = std::min(process_share, last_process_share); 
     
      MPI_SAFE_CALL(
          MPI_Scatter( 
              X.colptr(batch_index), /* start of matrix to send. */ // memptr() ?? 
              num_features * final_process_share, /* send length */
              MPI_DOUBLE,
              CPU_data.X, /* start of matrix to receive */
              num_features * final_process_share, /* receive length */
              MPI_DOUBLE,
              0, /* from root to other ps */
              MPI_COMM_WORLD
      ));


      MPI_SAFE_CALL(
          MPI_Scatter( 
              y.colptr(batch_index), /* start of matrix to send. */ // memptr() ?? 
              num_features * final_process_share, /* send length */
              MPI_DOUBLE,
              CPU_data.y, /* start of matrix to receive */
              num_features * final_process_share, /* receive length */
              MPI_DOUBLE,
              0, /* from root to other ps */
              MPI_COMM_WORLD
      ));


      /*
      2b. cudaMemcpy() CPU to GPU 
      
      GPU compute feedforward()
      GPU compute backprop()

      2c. cudaMemcpy() GPU to CPU       
      */
      checkCudaErrors(cudaMemcpy(GPU_data.X, CPU_data.X, sizeof(double) * num_features * final_process_share, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.y, CPU_data.y, sizeof(double) * num_classes * final_process_share, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.W1, nn.W[0].memptr(), sizeof(double) * num_features * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.b1, nn.b[0].memptr(), sizeof(double) * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.W2, nn.W[1].memptr(), sizeof(double) * num_neurons * num_classes, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.b2, nn.b[1].memptr(), sizeof(double) * num_classes, cudaMemcpyHostToDevice));

      gpu_feedforward(GPU_data, final_process_share, nn);

      gpu_backprop(GPU_data, final_process_share, reg, nn, num_procs);

      // GPU to CPU
      checkCudaErrors(cudaMemcpy(CPU_data.dW1, GPU_data.dW1, sizeof(double) * num_features * num_neurons, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(CPU_data.db1, GPU_data.db1, sizeof(double) * num_neurons, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(CPU_data.dW2, GPU_data.dW2, sizeof(double) * num_neurons * num_classes, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(CPU_data.db2, GPU_data.db2, sizeof(double) * num_classes, cudaMemcpyDeviceToHost));


      /* 
      2d. MPI_Allreduce() at local CPU 
      
      MPI_Allreduce()
      https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/ 
      https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node109.htm 
      */
      MPI_SAFE_CALL(
          // MPI_Allreduce == MPI_reduce + MPI_broadcast 
          // every node is a root
          MPI_Allreduce( 
              CPU_data.dW1, /* start addr of send buffer */
              CPU_data.dW1_reduced, /* start addr of receive buffer */
              (num_features * num_neurons), /* send count */
              MPI_DOUBLE,
              MPI_SUM, /* summing */
              MPI_COMM_WORLD
      ));
      MPI_SAFE_CALL(
          MPI_Allreduce(
              CPU_data.db1,
              CPU_data.db1_reduced,
              (num_neurons),
              MPI_DOUBLE,
              MPI_SUM,
              MPI_COMM_WORLD
      ));
      MPI_SAFE_CALL(
          MPI_Allreduce(
              CPU_data.dW2,
              CPU_data.dW2_reduced,
              (num_neurons * num_classes),
              MPI_DOUBLE,
              MPI_SUM,
              MPI_COMM_WORLD
      ));
      MPI_SAFE_CALL(
          MPI_Allreduce(
              CPU_data.db2,
              CPU_data.db2_reduced,
              (num_classes),
              MPI_DOUBLE,
              MPI_SUM,
              MPI_COMM_WORLD
      ));


      /* 
      Q: 4 GPU cores duplicately compute, then independently update each copy of nn
      == theoractically seems quicker than compute in CPU 
      or 1 core compute then MPI to other ps ?? 


      2e. cudaMemcpy() CPU to GPU 

      GPU compute gradient descent each time 4 

      2f. cudaMemcpy() GPU to CPU 
      */

      checkCudaErrors(cudaMemcpy(GPU_data.dW1, CPU_data.dW1_reduced, sizeof(double) * num_features * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.db1, CPU_data.db1_reduced, sizeof(double) * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.dW2, CPU_data.dW2_reduced, sizeof(double) * num_neurons * num_classes, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(GPU_data.db2, CPU_data.db2_reduced, sizeof(double) * num_classes, cudaMemcpyHostToDevice));

      gpu_gradientdescent(GPU_data, learning_rate);

      checkCudaErrors(cudaMemcpy(nn.W[0].memptr(), GPU_data.W1, sizeof(double) * num_features * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(nn.b[0].memptr(), GPU_data.b1, sizeof(double) * num_neurons, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(nn.W[1].memptr(), GPU_data.W2, sizeof(double) * num_neurons * num_classes, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(nn.b[1].memptr(), GPU_data.b2, sizeof(double) * num_classes, cudaMemcpyHostToDevice));


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && rank == 0 && print_flag) {
        // TODO
        // Copy data back to the CPU

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
        write_diff_gpu_cpu(nn, iter, error_file);
      }

      iter++;
    }
  }

  // TODO
  // Copy data back to the CPU

  error_file.close();

  // TODO
  // Free memory
}
