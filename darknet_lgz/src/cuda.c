int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
<<<<<<< HEAD
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
=======
#include "assert.h"
#include <stdlib.h>
#include <time.h>


void check_error(cudaError_t status)
{
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
<<<<<<< HEAD
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
=======
         x = ceil(sqrt(k));
         y = (n-1)/(x*BLOCK) + 1;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
<<<<<<< HEAD
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
=======
    static int init = 0;
    static cudnnHandle_t handle;
    if(!init) {
        cudnnCreate(&handle);
        init = 1;
    }
    return handle;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}
#endif

cublasHandle_t blas_handle()
{
<<<<<<< HEAD
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
=======
    static int init = 0;
    static cublasHandle_t handle;
    if(!init) {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
<<<<<<< HEAD
    } else {
        fill_ongpu(n, 0, x_gpu, 1);
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
<<<<<<< HEAD
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
=======
    static curandGenerator_t gen;
    static int init = 0;
    if(!init){
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(0));
        init = 1;
    }
    curandGenerateUniform(gen, x_gpu, n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

<<<<<<< HEAD
int *cuda_make_int_array(int *x, size_t n)
=======
int *cuda_make_int_array(size_t n)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
<<<<<<< HEAD
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

<<<<<<< HEAD
float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#endif
