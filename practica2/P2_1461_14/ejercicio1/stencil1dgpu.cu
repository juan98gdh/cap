#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define STENCIL_RADIUS 3

__global__ void stencil1D(int *in, int *out, int size, int stencil_radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // memoria compartida
    extern __shared__ int shared_data[];


    int shared_idx = threadIdx.x + stencil_radius;
    if (idx < size) {
        shared_data[shared_idx] = in[idx];
    }

    // Cargamos los datos de los bordes en memoria compartida
    if (threadIdx.x < stencil_radius) {
        int left = idx - stencil_radius;
        int right = idx + blockDim.x;
        if (left >= 0) {
            shared_data[shared_idx - stencil_radius] = in[left];
        }
        if (right < size) {
            shared_data[shared_idx + blockDim.x] = in[right];
        }
    }

    __syncthreads(); 

    if (idx < size) {
        int result = 0;
        for (int i = -stencil_radius; i <= stencil_radius; i++) {
            result += shared_data[shared_idx + i];
        }
        out[idx] = result / (2 * stencil_radius + 1);
    }
}


int main(int argc, char **argv) {
    //int N = 100000;
    int block_size = 256;

    if(argc != 2){
        printf("Introduce el tamaño del array como argumento.");
    }
    int N = atoi(argv[1]);

    int *input = (int *)malloc(N * sizeof(int));
    int *output = (int *)malloc(N * sizeof(int));

    // Inicializamos el array con valores aleatorios
    for (int i = 0; i < N; i++) {
        input[i] = rand() % 100;
    }

    // Reserva de memoria para la GPU
    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_output, N * sizeof(int));

    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N + block_size - 1) / block_size;


    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Lanzamos el kernel de GPU
    stencil1D<<<num_blocks, block_size, (block_size + 2 * STENCIL_RADIUS) * sizeof(int)>>>(d_input, d_output, N, STENCIL_RADIUS);

    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);

	  printf("%0.8f\n", (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec));

    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberamos memoria
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}
