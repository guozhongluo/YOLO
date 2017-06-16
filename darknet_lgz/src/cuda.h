#ifndef CUDA_H
#define CUDA_H

<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
extern int gpu_index;

#ifdef GPU

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
float *cuda_make_array(float *x, size_t n);
int *cuda_make_int_array(size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_free(float *x_gpu);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
