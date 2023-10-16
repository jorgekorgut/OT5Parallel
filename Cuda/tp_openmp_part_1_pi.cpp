/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

double calculatePi(int num_steps);
double calculatePartialPi(int a, int b);

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <algorithm>

static long num_steps = 1e8;
double step;

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
	  
    step = 1.0/(double) num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
    double pi = calculatePi(num_steps);
	  
    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);
}

double calculatePi(int num_steps)
{
  int num_threads = 64;
  int threadSize = 8;

  double sum = 0.0;
  int i;
  for (i=1;i<= num_steps; i+=threadSize){
		  sum += calculatePartialPi(i,std::min(i+threadSize,num_steps));
	}

  return step * sum;
}

double calculatePartialPi(int a, int b)
{
  double x, sum = 0;
  int i;
  for(i=a; i<b ; i++){
      x = (i-0.5)*step;
		  sum = sum + 4.0/(1.0+x*x);
  }
  return sum;
}