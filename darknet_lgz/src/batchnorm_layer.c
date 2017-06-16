<<<<<<< HEAD
<<<<<<< HEAD
#include "convolutional_layer.h"
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
<<<<<<< HEAD
<<<<<<< HEAD
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta  = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.scale_updates = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    l.bias_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = calloc(c, sizeof(float));
    l.variance = calloc(c, sizeof(float));

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
#ifdef GPU
    l.forward_gpu = forward_batchnorm_layer_gpu;
    l.backward_gpu = backward_batchnorm_layer_gpu;

    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);

    l.biases_gpu = cuda_make_array(l.biases, c);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

    l.scales_gpu = cuda_make_array(l.scales, c);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

    l.mean_gpu = cuda_make_array(l.mean, c);
    l.variance_gpu = cuda_make_array(l.variance, c);

    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
    l.rolling_variance_gpu = cuda_make_array(l.variance, c);

    l.mean_delta_gpu = cuda_make_array(l.mean, c);
    l.variance_delta_gpu = cuda_make_array(l.variance, c);

    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    #endif
#endif
    return l;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    layer layer = {0};
    layer.type = BATCHNORM;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.output = calloc(h * w * c * batch, sizeof(float));
    layer.delta  = calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.scales = calloc(c, sizeof(float));
    layer.scale_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        layer.scales[i] = 1;
    }

    layer.mean = calloc(c, sizeof(float));
    layer.variance = calloc(c, sizeof(float));

    layer.rolling_mean = calloc(c, sizeof(float));
    layer.rolling_variance = calloc(c, sizeof(float));
#ifdef GPU
    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);
    layer.delta_gpu =   cuda_make_array(layer.delta, h * w * c * batch);

    layer.scales_gpu = cuda_make_array(layer.scales, c);
    layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, c);

    layer.mean_gpu = cuda_make_array(layer.mean, c);
    layer.variance_gpu = cuda_make_array(layer.variance, c);

    layer.rolling_mean_gpu = cuda_make_array(layer.mean, c);
    layer.rolling_variance_gpu = cuda_make_array(layer.variance, c);

    layer.mean_delta_gpu = cuda_make_array(layer.mean, c);
    layer.variance_delta_gpu = cuda_make_array(layer.variance, c);

    layer.x_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
    layer.x_norm_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
#endif
    return layer;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
<<<<<<< HEAD
<<<<<<< HEAD
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
=======
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_batchnorm_layer(layer l, network net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_batchnorm_layer(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if(state.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);   
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);   
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
<<<<<<< HEAD
<<<<<<< HEAD
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

void backward_batchnorm_layer(const layer layer, network_state state)
{
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_batchnorm_layer_gpu(layer l, network net)
{
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                l.x_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#else
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_batchnorm_layer_gpu(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if (state.train) {
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .95, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .05, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .95, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .05, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
<<<<<<< HEAD
<<<<<<< HEAD

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_ongpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    }

    scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer_gpu(const layer l, network_state state)
{
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
<<<<<<< HEAD
<<<<<<< HEAD
#endif
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
=======
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}
#endif
