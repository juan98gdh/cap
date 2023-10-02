#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define ARRAY_SIZE 2048

/*
 * Statically allocate our arrays.  Compilers can
 * align them correctly.
 */
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c;

int main(int argc, char *argv[]) {
    int i,t;

    int number_of_trials = atoi(argv[1]);

    struct timeval start, stop;

    double m = 1.0001;

    /* Populate A and B arrays */
    for (i=0; i < ARRAY_SIZE; i++) {
        b[i] = i;
        a[i] = i+1;
    }

    gettimeofday(&start, NULL);

    /* Perform an operation a number of times */
    for (t=0; t < number_of_trials; t++) {
        for (i=0; i < ARRAY_SIZE; i++) {
            c += m*a[i] + b[i];
        }
    }

    gettimeofday(&stop, NULL);

    printf("%f\n", c);
    printf("Time: %0.6f\n", (stop.tv_sec - start.tv_sec) + 1e-6*(stop.tv_usec - start.tv_usec));


    return 0;
}
