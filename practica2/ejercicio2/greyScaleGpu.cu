#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define BLOCK_SIZE 256

__global__ void multiplyRatios(uint8_t *in, float *out, float* d_in_mul) {
    __shared__ uint8_t aux[BLOCK_SIZE];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    aux[lindex] = in[gindex] * d_in_mul[lindex];

    __syncthreads();

    out[gindex] = aux[lindex];
}

__global__ void rgbToGrey(float *in, uint8_t *out) {
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int gindex_temp = gindex * 4; 
  int result = 0;
  for (int offset = 0 ; offset < 4 ; offset++) {
    result += in[gindex_temp + offset];
  }

  out[gindex] = result;
}

int main(int nargs, char **argv)
{
    int width, height, nchannels;
    struct timeval fin,ini;
    uint8_t *d_in, *d_out;
    float *d_out_mul, *d_in_mul;
    float mul_ratios[BLOCK_SIZE];

    if (nargs < 2)
    {
        printf("Usage: %s <image1> [<image2> ...]\n", argv[0]);
    }
    // For each image
    // Bucle 0
    for (int file_i = 1; file_i < nargs; file_i++)
    {
        printf("[info] Processing %s\n", argv[file_i]);
        /****** Reading file ******/
        uint8_t *rgb_image = stbi_load(argv[file_i], &width, &height, &nchannels, 4);
        if (!rgb_image)
        {
            perror("Image could not be opened");
        }

        /****** Allocating memory ******/
        // - RGB2Grey
        uint8_t *grey_image = (uint8_t *) calloc(width, height);
        if (!grey_image)
        {
            perror("Could not allocate memory");
        }

        int size = width * height;

        cudaMalloc((void **) &d_in, size * 4 * sizeof(uint8_t));
        cudaMalloc((void **) &d_out, size * sizeof(uint8_t));
        cudaMalloc((void **) &d_out_mul, size * 4 * sizeof(float));         
        cudaMalloc((void **) &d_in_mul, BLOCK_SIZE * sizeof(float));
        cudaMemcpy(d_in, rgb_image, size * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice);         
        cudaMemcpy(d_out, grey_image, size * sizeof(uint8_t), cudaMemcpyHostToDevice); 
        
        //inicializamos el array de mul_ratios
        for (int i = 0; i < BLOCK_SIZE; i += 4) {
          mul_ratios[i] = 0.2989;
          mul_ratios[i + 1] = 0.5870;
          mul_ratios[i + 2] = 0.1140;
          mul_ratios[i + 3] = 0;
        }

        // - Filenames
        for (int i = strlen(argv[file_i]) - 1; i >= 0; i--)
        {
            if (argv[file_i][i] == '.')
            {
                argv[file_i][i] = 0;
                break;
            }
        }

        char *grey_image_filename = 0;
        asprintf(&grey_image_filename, "%s_grey.jpg", argv[file_i]);
        if (!grey_image_filename)
        {
            perror("Could not allocate memory");
            exit(-1);
        }

        /****** Computations ******/
        printf("[info] %s: width=%d, height=%d, nchannels=%d\n", argv[file_i], width, height, nchannels);

        if (nchannels != 3 && nchannels != 4)
        {
            printf("[error] Num of channels=%d not supported. Only three (RGB), four (RGBA) are supported.\n", nchannels);
            continue;
        }

        gettimeofday(&ini,NULL);

        cudaMemcpy(d_in_mul, mul_ratios, sizeof(mul_ratios), cudaMemcpyHostToDevice);                 
        multiplyRatios<<<(size*4)/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out_mul, d_in_mul);
        rgbToGrey<<<size/BLOCK_SIZE, BLOCK_SIZE>>>(d_out_mul, d_out);
        cudaMemcpy(grey_image, d_out, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);
        free(rgb_image);
        gettimeofday(&fin,NULL);

	      printf("Tiempo: %f\n", ((fin.tv_sec*1000000+fin.tv_usec)-(ini.tv_sec*1000000+ini.tv_usec))*1.0/1000000.0);
        free(grey_image_filename);
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_out_mul);
    }
}
