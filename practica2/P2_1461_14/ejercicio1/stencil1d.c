#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void stencil1D(int *in, int *out, int size) {
    for (int i = 1; i < size - 1; i++) {
        // Calcula el valor en out[i] como el promedio de los valores en in[i-1], in[i] y in[i+1]
        out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3;
    }

    // Trata los casos especiales de los bordes izquierdo y derecho
    out[0] = (in[0] + in[1]) / 2;
    out[size - 1] = (in[size - 2] + in[size - 1]) / 2;
}

int main(int argc, char **argv) {
    //int N = 100000;

    if(argc != 2){
        printf("Introduce el tamaño del array como argumento.");
    }
    int N = atoi(argv[1]);

    // Reservamos memoria para los arrays
    int *input = (int *)malloc(N * sizeof(int));
    int *output = (int *)malloc(N * sizeof(int));

    // inicializamos el array a valores aleatorios
    for (int i = 0; i < N; i++) {
        input[i] = rand() % 100;
    }

    // empezamos a medir el tiempo
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // llamada a stencil1d
    stencil1D(input, output, N);

    gettimeofday(&end, NULL);


	  printf("%0.8f\n", (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec));

    // liberamos memoria
    free(input);
    free(output);

    return 0;
}
