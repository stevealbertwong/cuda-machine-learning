#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;

typedef std::vector<int> vint;

#ifndef NDEBUG
#define LogInfo(M, ...) fprintf(stderr, "[INFO] (%s:%d:%d) " M "\n", __FILE__, __LINE__, omp_get_thread_num(), ##__VA_ARGS__)
#else
#define LogInfo(...)
#endif

void BitonicSortSeq(int start, int length, vint &seq, bool up);
void BitonicSortPar(int start, int length, vint &seq, bool up, int chunk);
void PrintSequence(int n, vint &seq, int s);

int main()
{
    // n must be a power of 2
#ifndef NDEBUG
    const int n = 1 << 4;
#else
    const int n = 1 << 23;
#endif
    printf("Size of array: %d\n", n);

    vint seq(n);
    std::generate(seq.begin(), seq.end(), [] { return rand() % 100; });

    vint seq_debug = seq;
    // Sort seq_debug
    sort(seq_debug.begin(), seq_debug.end());

    // print the original sequence
    PrintSequence(n, seq, n);

    // Use
    // $ export OMP_NUM_THREADS=4
    // to set the number of threads
    int n_threads;
#pragma omp parallel
#pragma omp single
    n_threads = omp_get_num_threads();

    // making sure input is okay
    if (n < n_threads * 2)
    {
        printf("The size of the sequence is less than 2 * the number of processes.\n");
        exit(0);
    }

    // the size of sub parts
    // chunk = n / n_threads;
    int chunk = 1;

    while (chunk <= (n / n_threads) >> 1)
    {
        chunk <<= 1;
    }

    assert(n_threads <= n / chunk);
    assert(n % chunk == 0);
    // n_threads <= n/chunk
    // chunk <= n/n_threads

    printf("Size of chunks: %d\n", chunk);
    printf("Number of chunks: %d\n", (n + chunk - 1) / chunk);
    printf("Number of threads: %d\n", n_threads);

    // start
    double begin_time, elapsed_time; /* for checking/testing timing */
    begin_time = omp_get_wtime();

    // This is the sequential algorithm
    // You need to turn this code into a parallel code
    // using #pragma omp parallel for
    // TODO
    for (int i = 2; i <= n; i <<= 1)
    {
        for (int j = 0; j < n; j += i)
        {
            bool up = ((j / i) % 2 == 0);
            BitonicSortSeq(j, i, seq, up);
        }

        LogInfo("Sort of length %d", i);
        PrintSequence(n, seq, i);
    }

    // end
    elapsed_time = omp_get_wtime() - begin_time;

    // print the sorted sequence
    LogInfo("Calculation complete");
    PrintSequence(n, seq, n);

    assert(equal(seq.begin(), seq.end(), seq_debug.begin()));

    LogInfo("Result of bitonic sort is correct and has passed the test.");

    printf("Elapsed time = %.2f sec, p T_p = %.2f.\n", elapsed_time,
           n_threads * elapsed_time);

    return 0;
}

void BitonicSortSeq(int start, int length, vint &seq, bool up)
{
    if (length == 1)
    {
        return;
    }

    if (length % 2 != 0)
    {
        printf("The length of a (sub)sequence is not divisible by 2.\n");
        exit(0);
    }

    const int split_length = length / 2;

    // bitonic split
    for (int i = start; i < start + split_length; i++)
    {
        if (up)
        {
            if (seq[i] > seq[i + split_length])
            {
                swap(seq[i], seq[i + split_length]);
            }
        }
        else if (seq[i] < seq[i + split_length])
        {
            swap(seq[i], seq[i + split_length]);
        }
    }

    // recursive sort
    BitonicSortSeq(start, split_length, seq, up);
    BitonicSortSeq(start + split_length, split_length, seq, up);
}

// Right now this routine is identical to BitonicSortSeq
// You need to change it to make it parallel.
// TODO
void BitonicSortPar(int start, int length, vint &seq, bool up, int chunk)
{
    if (length == 1)
    {
        return;
    }

    if (length % 2 != 0)
    {
        printf("The length of a (sub)sequence is not divisible by 2.\n");
        exit(0);
    }

    const int split_length = length / 2;

    // bitonic split
    // TODO
    for (int i = start; i < start + split_length; i++)
    {
        if (up)
        {
            if (seq[i] > seq[i + split_length])
            {
                swap(seq[i], seq[i + split_length]);
            }
        }
        else if (seq[i] < seq[i + split_length])
        {
            swap(seq[i], seq[i + split_length]);
        }
    }

    // TODO
    BitonicSortPar(start, split_length, seq, up, chunk);
    BitonicSortPar(start + split_length, split_length, seq, up, chunk);
}

void PrintSequence(int n, vint &seq, int s)
{
#ifndef NDEBUG
    int color = 0;
    printf("[");

    for (int i = 0; i < n; i++)
    {
        if (i % s == 0)
        {
            color = 1 - color;
        }

        if (color == 1)
        {
            printf("\x1B[0m%3d", seq[i]);
        }
        else
        {
            printf("\x1B[31m%3d", seq[i]);
        }
    }

    printf("\x1B[0m]\n");
#endif
}
