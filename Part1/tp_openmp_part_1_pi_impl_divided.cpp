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
#include <omp.h>
#include <iostream>

static long num_steps = 1e9;
static long num_cores = 12;
double step;

int main(int argc, char **argv)
{

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0))
    {
      num_steps = atol(argv[++i]);
      // printf( "  User num_steps is %ld\n", num_steps );
    }
    else if ((strcmp(argv[i], "-C") == 0) || (strcmp(argv[i], "-num_cores") == 0))
    {
      num_cores = atol(argv[++i]);
      // printf( "  User num_steps is %ld\n", num_steps );
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("  Pi Options:\n");
      printf("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  int i;
  double x, pi, sum, localSum = 0.0;
  int threadSize = 1e3;

  step = 1.0 / (double)num_steps;

  // Timer products.
  struct timeval begin, end;

  gettimeofday(&begin, NULL);

#pragma omp parallel for reduction(+ : sum) firstprivate(localSum) num_threads(num_cores)
  for (i = 1; i <= num_steps; i += threadSize)
  {
    localSum = 0;
    for (int j = i; j < i + threadSize && j <= num_steps; j++)
    {
      // std::cout << j << std::endl;
      x = (j - 0.5) * step;
      x = 4.0 / (1.0 + x * x);
      localSum = localSum + x;
    }

    sum += localSum;
  }

  pi = step * sum;

  gettimeofday(&end, NULL);

  // Calculate time.
  double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cerr << pi << std::endl;
  printf("%lf", time);
}
