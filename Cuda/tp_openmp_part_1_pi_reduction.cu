/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <algorithm>
#include <iostream>

__global__ void calculatePi(double *pi, double step, int num_steps, int threadSize, int num_threads);

__device__ double calculatePartialPi(int a, int b, double step, int num_steps);

static long num_steps = 1e8;

int main(int argc, char **argv)
{

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0))
    {
      num_steps = atol(argv[++i]);
      std::cerr << "  User num_steps is" << num_steps << std::endl;
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      std::cerr << ("  Pi Options:\n");
      std::cerr << ("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
      std::cerr << ("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  double step = 1.0 / (double)num_steps;

  // Timer products.
  struct timeval begin, end;

  gettimeofday(&begin, NULL);

  int size = sizeof(double);
  double *h_sum = (double *)malloc(size);
  *h_sum = 0;
  double *d_sum;
  cudaMalloc(&d_sum, size);
  cudaMemcpy(d_sum, h_sum, size, cudaMemcpyHostToDevice);

  int num_blocks = 1024;
  int threadSize = num_steps / num_blocks + 1;
  int num_threads = 4;

  calculatePi<<<num_blocks, num_threads, num_threads>>>(d_sum, step, num_steps, threadSize, num_threads);

  // cudaDeviceSynchronize();
  cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

  double pi = step * (*h_sum);

  cudaFree(d_sum);
  free(h_sum);

  gettimeofday(&end, NULL);

  // Calculate time.
  double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cerr << "pi with: " << num_steps << std::endl;
  std::cout << time;
}

__global__ void calculatePi(double *sum, double step, int num_steps, int threadSize, int num_threads)
{
  int i = (blockIdx.x * num_threads + threadIdx.x) * threadSize + 1;

  double partialSum = calculatePartialPi(i, i + threadSize, step, num_steps);

  atomicAdd(sum, partialSum);
}

__device__ double calculatePartialPi(int a, int b, double step, int num_steps)
{

  double x, sum = 0;
  int i;
  for (i = a; i < b && i < num_steps; i++)
  {
    x = (i - 0.5) * step;

    sum = sum + 4.0 / (1.0 + x * x);
  }

  return sum;
}