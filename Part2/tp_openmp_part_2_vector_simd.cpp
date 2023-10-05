/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <cmath>
#include <float.h>
#include <omp.h>

void checkSizes(int &N, int &M, int &S, int &nrepeat);

int main(int argc, char *argv[])
{
  int N = -1;        // number of rows 2^12
  int M = -1;        // number of columns 2^10
  int S = -1;        // total size 2^22
  int nrepeat = 1;   // number of repeats of the test
  int num_cores = 1; // number of cores

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0))
    {
      N = pow(2, atoi(argv[++i]));
      fprintf(stderr, "  User N is %d\n", N);
    }
    else if ((strcmp(argv[i], "-M") == 0) || (strcmp(argv[i], "-Columns") == 0))
    {
      M = pow(2, atof(argv[++i]));
      fprintf(stderr, "  User M is %d\n", M);
    }
    else if ((strcmp(argv[i], "-S") == 0) || (strcmp(argv[i], "-Size") == 0))
    {
      S = pow(2, atof(argv[++i]));
      fprintf(stderr, "  User S is %d\n", S);
    }
    else if (strcmp(argv[i], "-nrepeat") == 0)
    {
      nrepeat = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "-C") == 0)
    {
      num_cores = atoi(argv[++i]);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      fprintf(stderr, "  y^T*A*x Options:\n");
      fprintf(stderr, "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n");
      fprintf(stderr, "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n");
      fprintf(stderr, "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n");
      fprintf(stderr, "  -nrepeat <int>:        number of repetitions (default: 100)\n");
      fprintf(stderr, "  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // Check sizes.
  checkSizes(N, M, S, nrepeat);

// Allocate x,y,A
#define VariableType float
#define VariableTypePlus float

  // N : number of rows
  auto y = new VariableTypePlus[N];
  // N : number of Columns
  auto x = new VariableType[M];
  auto A = new VariableType[N * M];

  // Initialize y vector to 1.
  for (int i = 0; i < N; i++)
  {
    y[i] = 1;
  }

  // Initialize x column vector to 1 .
  for (int i = 0; i < M; i++)
  {
    x[i] = 1;
  }

// Initialize A matrix, you can use a 1D index if you want a flat structure (i.e. a 1D array) e.g. j*M+i is the same than [j][i]
#pragma omp parallel for num_threads(num_cores)
  for (int i = 0; i < N * M; i++)
  {
    A[i] = 1;
  }

  // Timer products.
  struct timeval begin, end;

  gettimeofday(&begin, NULL);
  VariableTypePlus sum;

  for (int repeat = 0; repeat < nrepeat; repeat++)
  {
    #pragma omp parallel for num_threads(num_cores)
    for (int i = 0; i < N; i++)
    {
      sum = 0;
      
      #pragma omp simd
      for (int j = 0; j < M; j++)
      {
        sum += A[i * M + j] * x[j];
        // std::cout << x[j] << std::endl;
      }
      y[i] = sum;
    }
    
    VariableTypePlus result = 0;
    #pragma omp parallel for reduction(+:result) num_threads(num_cores)
    for (int i = 0; i < N; i++)
    {
      result += y[i];
    }

    // Output result.
    if (repeat == (nrepeat - 1))
    {
      fprintf(stderr, "  Computed result for %d x %d is %lf\n", N, M, result);
    }

    const double solution = (double)N * (double)M;

    if (result != solution)
    {
      fprintf(stderr, "  Error: result( %lf ) != solution( %lf )\n", result, solution);
    }
  }

  gettimeofday(&end, NULL);

  // Calculate time.
  // double time = timer.seconds();
  double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

  // Print results (problem size, time and bandwidth in GB/s).
  fprintf(stderr, "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);

  std::cout << time;

  std::free(A);
  std::free(y);
  std::free(x);

  return 0;
}

void checkSizes(int &N, int &M, int &S, int &nrepeat)
{
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if (S == -1 && (N == -1 || M == -1))
  {
    S = pow(2, 22);
    if (S < N)
      S = N;
    if (S < M)
      S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if (S == -1)
    S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if (N == -1 && M == -1)
  {
    if (S > 1024)
    {
      M = 1024;
    }
    else
    {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if (M == -1)
    M = S / N;

  // If N is undefined, set it.
  if (N == -1)
    N = S / M;

  fprintf(stderr, "  Total size S = %d N = %d M = %d\n", S, N, M);

  // Check sizes.
  if ((S < 0) || (N < 0) || (M < 0) || (nrepeat < 0))
  {
    fprintf(stderr, "  Sizes must be greater than 0.\n");
    exit(1);
  }

  if ((N * M) != S)
  {
    fprintf(stderr, "  N * M != S\n");
    exit(1);
  }
}
