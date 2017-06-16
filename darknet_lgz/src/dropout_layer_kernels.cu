#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
=======
void forward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if (!state.train) return;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
void forward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if (!state.train) return;
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

<<<<<<< HEAD
<<<<<<< HEAD
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.input, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if(!state.delta) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.delta, size, layer.rand_gpu, layer.probability, layer.scale);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    check_error(cudaPeekAtLastError());
}
