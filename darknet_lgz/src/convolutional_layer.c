#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

<<<<<<< HEAD
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
=======
#ifndef AI2
#define AI2 0
//void forward_xnor_layer(layer l, network_state state);
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->filters;
    l->filters = l->binary_filters;
    l->binary_filters = swap;

    #ifdef GPU
    swap = l->filters_gpu;
    l->filters_gpu = l->binary_filters_gpu;
    l->binary_filters_gpu = swap;
    #endif
}

void binarize_filters(float *filters, int n, int size, float *binary)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
<<<<<<< HEAD
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
=======
            mean += fabs(filters[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
<<<<<<< HEAD
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
=======
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

int convolutional_out_width(convolutional_layer l)
{
<<<<<<< HEAD
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
=======
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

image get_convolutional_image(convolutional_layer l)
{
<<<<<<< HEAD
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
=======
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

image get_convolutional_delta(convolutional_layer l)
{
<<<<<<< HEAD
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
=======
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

size_t get_workspace_size(layer l){
#ifdef CUDNN
    size_t most = 0;
    size_t s = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.filterDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dfilterDesc,
            l.bf_algo,
            &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.filterDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dsrcTensorDesc,
            l.bd_algo,
            &s);
    if (s > most) most = s;
    return most;
#else
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
#endif
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
<<<<<<< HEAD
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
=======
    cudnnSetFilter4dDescriptor(l->dfilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    int padding = l->pad ? l->size/2 : 0;
    cudnnSetConvolution2dDescriptor(l->convDesc, padding, padding, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->filterDesc,
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
<<<<<<< HEAD
            l->weightDesc,
=======
            l->filterDesc,
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
<<<<<<< HEAD
            l->dweightDesc,
=======
            l->dfilterDesc,
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif

<<<<<<< HEAD
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
=======
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize, int binary, int xnor)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
<<<<<<< HEAD
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));
=======
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.filters = calloc(c*n*size*size, sizeof(float));
    l.filter_updates = calloc(c*n*size*size, sizeof(float));
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

<<<<<<< HEAD
    l.nweights = c*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
=======
    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.filters[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

<<<<<<< HEAD
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
=======
    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));

    if(binary){
        l.binary_filters = calloc(c*n*size*size, sizeof(float));
        l.cfilters = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_filters = calloc(c*n*size*size, sizeof(float));
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

<<<<<<< HEAD
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
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

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

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
=======
        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    l.filter_updates_gpu = cuda_make_array(l.filter_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.scales_gpu = cuda_make_array(l.scales, n);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    if(binary){
        l.binary_filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    }
    if(xnor){
        l.binary_filters_gpu = cuda_make_array(l.filters, c*n*size*size);
        l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, n);
        l.variance_gpu = cuda_make_array(l.variance, n);

        l.rolling_mean_gpu = cuda_make_array(l.mean, n);
        l.rolling_variance_gpu = cuda_make_array(l.variance, n);

        l.mean_delta_gpu = cuda_make_array(l.mean, n);
        l.variance_delta_gpu = cuda_make_array(l.variance, n);

        l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    }
#ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.srcTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnCreateFilterDescriptor(&l.filterDesc);
    cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
    cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
    cudnnCreateFilterDescriptor(&l.dfilterDesc);
    cudnnCreateConvolutionDescriptor(&l.convDesc);
    cudnn_convolutional_setup(&l);
#endif
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

<<<<<<< HEAD
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
=======
    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
<<<<<<< HEAD
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
=======
            l.filters[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
<<<<<<< HEAD
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/
=======
    network_state state = {0};
    state.input = data;
    forward_convolutional_layer(l, state);
}
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

<<<<<<< HEAD
    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }
=======
    l->output = realloc(l->output,
            l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta  = realloc(l->delta,
            l->batch*out_h * out_w * l->n*sizeof(float));
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

<<<<<<< HEAD
    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
=======
    l->delta_gpu =     cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
    l->output_gpu =    cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

<<<<<<< HEAD
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
=======
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;


    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    /*
       if(l.binary){
       binarize_filters(l.filters, l.n, l.c*l.size*l.size, l.binary_filters);
       binarize_filters2(l.filters, l.n, l.c*l.size*l.size, l.cfilters, l.scales);
       swap_binary(&l);
       }
     */

    /*
       if(l.binary){
       int m = l.n;
       int k = l.size*l.size*l.c;
       int n = out_h*out_w;

       char  *a = l.cfilters;
       float *b = state.workspace;
       float *c = l.output;

       for(i = 0; i < l.batch; ++i){
       im2col_cpu(state.input, l.c, l.h, l.w, 
       l.size, l.stride, l.pad, b);
       gemm_bin(m,n,k,1,a,k,b,n,c,n);
       c += n*m;
       state.input += l.c*l.h*l.w;
       }
       scale_bias(l.output, l.scales, l.batch, l.n, out_h*out_w);
       add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
       activate_array(l.output, m*n*l.batch, l.activation);
       return;
       }
     */

    if(l.xnor ){
        binarize_filters(l.filters, l.n, l.c*l.size*l.size, l.binary_filters);
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;

<<<<<<< HEAD

    float *a = l.weights;
    float *b = net.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(net.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        net.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }
=======
    if (l.xnor && l.c%32 == 0 && AI2) {
        //forward_xnor_layer(l, state);
        printf("xnor\n");
    } else {

        float *a = l.filters;
        float *b = state.workspace;
        float *c = l.output;

        for(i = 0; i < l.batch; ++i){
            im2col_cpu(state.input, l.c, l.h, l.w, 
                    l.size, l.stride, l.pad, b);
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            c += n*m;
            state.input += l.c*l.h*l.w;
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

<<<<<<< HEAD
void backward_convolutional_layer(convolutional_layer l, network net)
=======
void backward_convolutional_layer(convolutional_layer l, network_state state)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
<<<<<<< HEAD
    int k = l.out_w*l.out_h;

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im = net.input+i*l.c*l.h*l.w;
=======
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = state.workspace;
        float *c = l.filter_updates;

        float *im = state.input+i*l.c*l.h*l.w;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

<<<<<<< HEAD
        if(net.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
=======
        if(state.delta){
            a = l.filters;
            b = l.delta + i*m*k;
            c = state.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

<<<<<<< HEAD
    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
=======
    axpy_cpu(size, -decay*batch, l.filters, 1, l.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.filter_updates, 1, l.filters, 1);
    scal_cpu(size, momentum, l.filter_updates, 1);
}


image get_convolutional_filter(convolutional_layer l, int i)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
<<<<<<< HEAD
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
=======
    return float_to_image(w,h,c,l.filters+i*h*w*c);
}

void rgbgr_filters(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_filter(l, i);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

<<<<<<< HEAD
void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
=======
void rescale_filters(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_filter(l, i);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

<<<<<<< HEAD
image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);
=======
image *get_filters(convolutional_layer l)
{
    image *filters = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        filters[i] = copy_image(get_convolutional_filter(l, i));
        //normalize_image(filters[i]);
    }
    return filters;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_filters)
{
    image *single_filters = get_filters(l);
    show_images(single_filters, l.n, window);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
<<<<<<< HEAD
    return single_weights;
=======
    return single_filters;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
}

