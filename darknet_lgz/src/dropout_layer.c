#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
<<<<<<< HEAD
<<<<<<< HEAD
=======
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
<<<<<<< HEAD
<<<<<<< HEAD
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
=======
    #ifdef GPU
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    #ifdef GPU
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if (!state.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) state.input[i] = 0;
        else state.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if(!state.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) state.delta[i] = 0;
        else state.delta[i] *= l.scale;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    }
}

