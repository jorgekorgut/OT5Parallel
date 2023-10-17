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

__global__
void calculatePi(double* pi, double step, int num_steps, int threadSize, int num_threads);

__device__ 
double calculatePartialPi(int a, int b, double step, int num_steps);

static long num_steps = 1e8;

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
	  
    double step = 1.0/(double) num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
    int size =sizeof(double);
    double * h_sum = (double*)malloc(size);
    *h_sum = 0;
    double * d_sum;
    cudaMalloc(&d_sum, size);
    cudaMemcpy(d_sum, h_sum, size, cudaMemcpyHostToDevice);

    int num_blocks = 1024;
    
    int num_threads = 32;

    int numStepsInThread = num_steps/num_blocks/num_threads + 1;

    calculatePi<<<num_blocks,num_threads,num_threads * sizeof(float)>>>(d_sum, step, num_steps, numStepsInThread, num_threads);

    //cudaDeviceSynchronize();
    cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

    double pi = step * (*h_sum);
	  
    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);
}

__global__
void calculatePi(double* sum, double step, int num_steps, int numStepsInThread, int num_threads)
{
  //int numberOfThreadInGrid = blockDim.x * gridDim.x;
  //int numberOfThreadsInBlock = blockDim.x;
  extern __shared__ double local_sum[];

  int i = (blockIdx.x*num_threads + threadIdx.x) * numStepsInThread + 1;
  
  double partialSum = calculatePartialPi(i,i+numStepsInThread, step, num_steps);

  local_sum[threadIdx.x] = partialSum;

  __syncthreads();

  //Reduction of blocks
  for(int j = 1; j < blockDim.x; j*=2){
    if(threadIdx.x%(2*j)==0){
      local_sum[threadIdx.x] += local_sum[threadIdx.x + j];
    }
    
    __syncthreads();
  }

  if(threadIdx.x ==0){
    printf("%lf\n", local_sum[0]);
    atomicAdd(sum, local_sum[0]);
  }
}

__device__  
double calculatePartialPi(int a, int b, double step, int num_steps)
{
  
  double x, sum = 0;
  int i;
  for(i=a; i<b && i < num_steps ; i++){
      x = (i-0.5)*step;
      
		  sum = sum + 4.0/(1.0+x*x);
  }
  
  return sum;
}