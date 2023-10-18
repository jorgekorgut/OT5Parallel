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

#include <memory>
#include <iostream>

__global__
void calculatePi(double* pi, double * block_sum, double step, int num_steps, int threadSize);

__device__ 
double calculatePartialPi(int a, int b, double step, int num_steps);

void printDeviceProps();

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
    
    //printDeviceProps();
    
    gettimeofday( &begin, NULL );
    
    int num_blocks = 1024;
    
    int num_threads = 256;

    int doubleSize =sizeof(double);
    double * h_sum = (double*)malloc(doubleSize);
    *h_sum = 0;
    double * d_sum;
    double * d_block_sum;

    cudaMalloc(&d_sum, doubleSize);
    cudaMalloc(&d_block_sum, doubleSize*num_blocks);
    cudaMemcpy(d_sum, h_sum, doubleSize, cudaMemcpyHostToDevice);

    int numStepsInThread = num_steps/num_blocks/num_threads + 1;

    calculatePi<<<num_blocks,num_threads,num_threads>>>(d_sum,d_block_sum, step, num_steps, numStepsInThread);
    
    cudaMemcpy(h_sum, d_sum, doubleSize, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    const char * errorName = cudaGetErrorName(error);
    const char * errorDescription = cudaGetErrorString(error);

    //printf("%s : %s",errorName, errorDescription);

    double pi = step * (*h_sum);
	  
    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);
}

void printDeviceProps()
{
  int runtimeVersion = 0;
  int driverVersion = 0;
  int dev = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
/*
        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        */
               printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

               printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");

              }

__global__
void calculatePi(double* sum, double * block_sum, double step, int num_steps, int numStepsInThread)
{
  //int numberOfThreadInGrid = blockDim.x * gridDim.x;
  int numberOfThreadsInBlock = blockDim.x;
  extern __shared__ double thread_sum[256];

  int i = (blockIdx.x*numberOfThreadsInBlock + threadIdx.x) * numStepsInThread + 1;
  
  double partialSum = calculatePartialPi(i,i+numStepsInThread, step, num_steps);

  thread_sum[threadIdx.x] = partialSum;

  __syncthreads();

  //Reduction of blocks
  for(int j = 1; j < blockDim.x; j*=2){
    if(threadIdx.x%(2*j)==0){
      thread_sum[threadIdx.x] += thread_sum[threadIdx.x + j];
    }
    
    __syncthreads();
  }

  if(threadIdx.x == 0){
    block_sum[blockIdx.x] = thread_sum[0];
    //printf("%lf\n", block_sum[blockIdx.x]);
  }
  __syncthreads();

  for(int j = 1; j < gridDim.x; j*=2){
    if(threadIdx.x == 0){
      if(blockIdx.x%(2*j)==0){
        block_sum[blockIdx.x] += block_sum[blockIdx.x + j];
      }
    }
    //printf("b: %d - t:%d \n", blockIdx.x, threadIdx.x);
    __syncthreads();
  }
  __syncthreads();
  if(threadIdx.x ==0 && 
     blockIdx.x ==0){
      *sum = 10;
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