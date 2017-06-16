#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
<<<<<<< HEAD
#include "batchnorm_layer.h"
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

<<<<<<< HEAD
extern "C" void forward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights_gpu;
        float *b = net.input_gpu + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.outputs);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_ongpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = net.input_gpu + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates_gpu;

        im2col_ongpu(l.delta_gpu + i*l.outputs, l.out_c, l.out_h, l.out_w, 
                l.size, l.stride, l.pad, b);
        gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta_gpu){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights_gpu;
            float *b = net.workspace;
            float *c = net.delta_gpu + i*n*m;

            gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
=======
extern "C" void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    fill_ongpu(layer.outputs*layer.batch, 0, layer.output_gpu, 1);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.filters_gpu;
        float *b = state.input + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image_gpu;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output_gpu+i*layer.n*size);
    }
    add_bias_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, size);
    activate_array(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;

    gradient_array(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias(layer.bias_updates_gpu, layer.delta, layer.batch, layer.n, size);

    if(state.delta) memset(state.delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = state.input + i*m*n;
        float *b = layer.col_image_gpu;
        float *c = layer.filter_updates_gpu;

        im2col_ongpu(layer.delta_gpu + i*layer.n*size, layer.n, out_h, out_w, 
                layer.size, layer.stride, 0, b);
        gemm_ongpu(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.filters_gpu;
            float *b = layer.col_image_gpu;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        }
    }
}

<<<<<<< HEAD
extern "C" void pull_deconvolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

extern "C" void push_deconvolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_deconvolutional_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;

    if(l.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, size, batch, l.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        }
    }else{
        axpy_ongpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_ongpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_ongpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_ongpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
=======
extern "C" void pull_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void push_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void update_deconvolutional_layer_gpu(deconvolutional_layer layer, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -decay, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

