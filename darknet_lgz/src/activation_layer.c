#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

<<<<<<< HEAD
<<<<<<< HEAD
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

=======
#ifdef GPU
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
#ifdef GPU
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_activation_layer(layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

#ifdef GPU

<<<<<<< HEAD
<<<<<<< HEAD
void forward_activation_layer_gpu(layer l, network net)
{
    copy_ongpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_activation_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}
#endif
