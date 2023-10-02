#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static inline void getRGB(uint8_t *im, int width, int height, int nchannels, int x, int y, int *r, int *g, int *b)
{

    unsigned char *offset = im + (x + width * y) * nchannels;
    *r = offset[0];
    *g = offset[1];
    *b = offset[2];
}

int main(int nargs, char **argv)
{
    int width, height, nchannels;
    struct timeval fin,ini;

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
        uint8_t *grey_image = malloc(width * height);
        if (!grey_image)
        {
            perror("Could not allocate memory");
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
        // RGB to grey scale
        int imageSize = width * height;

        for(int i = 0, j = 0; j < imageSize; i += 16, j += 4) {
            
            __m128i* data_ptr_one = (__m128i*)(rgb_image + i);
            __m128i* data_ptr_two = (__m128i*)(rgb_image + i + 8);
            __m128i filas = _mm_loadl_epi64(data_ptr_one); 
            __m128i columnas = _mm_loadl_epi64(data_ptr_two);

            // Extendemos los vectores y los convertimos a floats
            __m256i extendedFilasInt = _mm256_cvtepu8_epi32(filas);
            __m256 extendedFilasFloat = _mm256_cvtepi32_ps(extendedFilasInt);
            __m256i extendedColumnasInt = _mm256_cvtepu8_epi32(columnas);
            __m256 extendedColumnasFloat = _mm256_cvtepi32_ps(extendedColumnasInt);

            // generamos el vector de coeficientes y lo usamos para multiplicar los vectores por pares de pixeles.
            __m256 coeficientes = _mm256_set_ps(0.0, 0.1140, 0.5870, 0.2989, 0.0, 0.1140, 0.5870, 0.2989);
            __m256 par1 = _mm256_mul_ps(extendedFilasFloat, coeficientes);
            __m256 par2 = _mm256_mul_ps(extendedColumnasFloat, coeficientes); 

            // hacemos un horizontal add, dos  veces porque no se completa en una sola.
            __m256 h_add = _mm256_hadd_ps(par1, par2);
            h_add = _mm256_hadd_ps(h_add, h_add);

            // permutamos el vector final y lo extraemos para el outcome.
            __m256 permutado = _mm256_permutevar8x32_ps(h_add, _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0)); // After permutations
            __m128 outcome = _mm256_extractf128_ps(permutado, 0);

            for (int k = 0; k < 4; k++) {
                grey_image[j + k] = (int) outcome[k];
            }
        }

        stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);
        free(rgb_image);

        gettimeofday(&fin,NULL);

	    printf("Tiempo: %f\n", ((fin.tv_sec*1000000+fin.tv_usec)-(ini.tv_sec*1000000+ini.tv_usec))*1.0/1000000.0);
        free(grey_image_filename);
    }
}
