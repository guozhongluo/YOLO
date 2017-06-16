#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
<<<<<<< HEAD
    if (strcmp(s, "L1")==0) return L1;
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
<<<<<<< HEAD
        case L1:
            return "L1";
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }
    return "sse";
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
<<<<<<< HEAD
    fprintf(stderr, "cost                                           %4d\n",  inputs);
=======
    fprintf(stderr, "Cost Layer: %d inputs\n", inputs);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    cost_layer l = {0};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));
<<<<<<< HEAD

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;
    #ifdef GPU
    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

=======
    #ifdef GPU
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
#endif
}

<<<<<<< HEAD
void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
=======
void forward_cost_layer(cost_layer l, network_state state)
{
    if (!state.truth) return;
    if(l.cost_type == MASKED){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(state.truth[i] == SECRET_NUM) state.input[i] = SECRET_NUM;
        }
    }
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, state.input, state.truth, l.delta, l.output);
    } else {
        l2_cpu(l.batch*l.inputs, state.input, state.truth, l.delta, l.output);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

<<<<<<< HEAD
void backward_cost_layer(const cost_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
=======
void backward_cost_layer(const cost_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, state.delta, 1);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

#ifdef GPU

void pull_cost_layer(cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

<<<<<<< HEAD
int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(cost_layer l, network net)
{
    if (!net.truth_gpu) return;
    if(l.smooth){
        scal_ongpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
        add_ongpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
    }
    if (l.cost_type == MASKED) {
        mask_ongpu(l.batch*l.inputs, net.input_gpu, SECRET_NUM, net.truth_gpu);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if(l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_ongpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        supp_ongpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
=======
void forward_cost_layer_gpu(cost_layer l, network_state state)
{
    if (!state.truth) return;
    if (l.cost_type == MASKED) {
        mask_ongpu(l.batch*l.inputs, state.input, SECRET_NUM, state.truth);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, state.input, state.truth, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, state.input, state.truth, l.delta_gpu, l.output_gpu);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

<<<<<<< HEAD
void backward_cost_layer_gpu(const cost_layer l, network net)
{
    axpy_ongpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
=======
void backward_cost_layer_gpu(const cost_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, state.delta, 1);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}
#endif

