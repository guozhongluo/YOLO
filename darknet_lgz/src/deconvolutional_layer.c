#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
<<<<<<< HEAD
#include "batchnorm_layer.h"
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
<<<<<<< HEAD

#include <stdio.h>
#include <time.h>


static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}


layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
=======
#include <stdio.h>
#include <time.h>

int deconvolutional_out_height(deconvolutional_layer l)
{
    int h = l.stride*(l.h - 1) + l.size;
    return h;
}

int deconvolutional_out_width(deconvolutional_layer l)
{
    int w = l.stride*(l.w - 1) + l.size;
    return w;
}

int deconvolutional_out_size(deconvolutional_layer l)
{
    return deconvolutional_out_height(l) * deconvolutional_out_width(l);
}

image get_deconvolutional_image(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_deconvolutional_delta(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

deconvolutional_layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    deconvolutional_layer l = {0};
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

<<<<<<< HEAD
    l.nweights = c*n*size*size;
    l.nbiases = n;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
=======
    l.filters = calloc(c*n*size*size, sizeof(float));
    l.filter_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    float scale = 1./sqrt(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) l.filters[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        l.biases[i] = scale;
    }
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);

    l.out_h = out_h;
    l.out_w = out_w;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

<<<<<<< HEAD
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    l.update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0){

        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
        }
    }
    #ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
    #endif
#endif

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
=======
    l.col_image = calloc(h*w*size*size*n, sizeof(float));
    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));

    #ifdef GPU
    l.filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    l.filter_updates_gpu = cuda_make_array(l.filter_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.col_image_gpu = cuda_make_array(l.col_image, h*w*size*size*n);
    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    #endif

    l.activation = activation;

    fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    return l;
}

<<<<<<< HEAD
void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
    #ifdef CUDNN
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    #endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void forward_deconvolutional_layer(const layer l, network net)
{
    int i;
=======
void resize_deconvolutional_layer(deconvolutional_layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    int out_h = deconvolutional_out_height(*l);
    int out_w = deconvolutional_out_width(*l);

    l->col_image = realloc(l->col_image,
                                out_h*out_w*l->size*l->size*l->c*sizeof(float));
    l->output = realloc(l->output,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta  = realloc(l->delta,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    #ifdef GPU
    cuda_free(l->col_image_gpu);
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->col_image_gpu = cuda_make_array(l->col_image, out_h*out_w*l->size*l->size*l->c);
    l->delta_gpu = cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
    l->output_gpu = cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
    #endif
}

void forward_deconvolutional_layer(const deconvolutional_layer l, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i){
<<<<<<< HEAD
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer(layer l, network net)
{
    int i;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));
=======
        float *a = l.filters;
        float *b = state.input + i*l.c*l.h*l.w;
        float *c = l.col_image;

        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output+i*l.n*size);
    }
    add_bias(l.output, l.biases, l.batch, l.n, size);
    activate_array(l.output, l.batch*l.n*size, l.activation);
}

void backward_deconvolutional_layer(deconvolutional_layer l, network_state state)
{
    float alpha = 1./l.batch;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;
    int i;

    gradient_array(l.output, size*l.n*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, size);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

<<<<<<< HEAD
        float *a = net.input + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.outputs, l.out_c, l.out_h, l.out_w, 
                l.size, l.stride, l.pad, b);
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta){
=======
        float *a = state.input + i*m*n;
        float *b = l.col_image;
        float *c = l.filter_updates;

        im2col_cpu(l.delta + i*l.n*size, l.n, out_h, out_w, 
                l.size, l.stride, 0, b);
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

<<<<<<< HEAD
            float *a = l.weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;

            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
=======
            float *a = l.filters;
            float *b = l.col_image;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        }
    }
}

<<<<<<< HEAD
void update_deconvolutional_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
=======
void update_deconvolutional_layer(deconvolutional_layer l, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay, l.filters, 1, l.filter_updates, 1);
    axpy_cpu(size, learning_rate, l.filter_updates, 1, l.filters, 1);
    scal_cpu(size, momentum, l.filter_updates, 1);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}



