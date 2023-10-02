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

            // Storing 2 pixels in each vector
            __m128i datal = _mm_loadl_epi64(rgb_image + i); // 2 pixeles
            __m128i datah = _mm_loadl_epi64(rgb_image + i + 8);


            // Extending each vector and converting to float
            __m256i extendedlInt = _mm256_cvtepu8_epi32(datal);
            __m256 extendedlFloat = _mm256_cvtepi32_ps(extendedlInt);

            __m256i extendedhInt = _mm256_cvtepu8_epi32(datah);
            __m256 extendedhFloat = _mm256_cvtepi32_ps(extendedhInt);

            // Generating the coefficients vector
            __m256 coefficients = _mm256_set_ps(0.0, 0.1140, 0.5870, 0.2989, 0.0, 0.1140, 0.5870, 0.2989);

            // Multiplicating each vector with the coefficients
            __m256 mull = _mm256_mul_ps(extendedlFloat, coefficients); // Pixel 1 and Pixel 2
            __m256 mulh = _mm256_mul_ps(extendedhFloat, coefficients); // Pixel 3 and Pixel 4

            // Now we can add horizontal add each vector
            __m256 hadded = _mm256_hadd_ps(mull, mulh);

            // The adding is not complete so we add a second time
            __m256 secondHadded = _mm256_hadd_ps(hadded, hadded);

            // The final vector has duplicates and a strange order so we permutate it
            __m256 reorder = _mm256_permutevar8x32_ps(secondHadded, _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0)); // After permutations

            // Extract the 128bit vector
            __m128 extracted = _mm256_extractf128_ps(reorder, 0);


            // Store the 4 pixels
            for (int k = 0; k < 4; k++) {
                grey_image[j + k] = (int) extracted[k];
            }
        }

        stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);
        free(rgb_image);

        gettimeofday(&fin,NULL);

	    printf("Tiempo: %f\n", ((fin.tv_sec*1000000+fin.tv_usec)-(ini.tv_sec*1000000+ini.tv_usec))*1.0/1000000.0);
        free(grey_image_filename);
    }
}
