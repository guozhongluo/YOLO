#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
<<<<<<< HEAD
<<<<<<< HEAD
    binary[i] = (x[i] >= 0) ? 1 : -1;
=======
    binary[i] = (x[i] > 0) ? 1 : -1;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    binary[i] = (x[i] > 0) ? 1 : -1;
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


<<<<<<< HEAD
<<<<<<< HEAD
__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
=======
__global__ void binarize_filters_kernel(float *filters, int n, int size, float *binary)
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
__global__ void binarize_filters_kernel(float *filters, int n, int size, float *binary)
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
<<<<<<< HEAD
<<<<<<< HEAD
        mean += abs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        mean += abs(filters[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;
    }
}

void binarize_filters_gpu(float *filters, int n, int size, float *binary)
{
    binarize_filters_kernel<<<cuda_gridsize(n), BLOCK>>>(filters, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        swap_binary(&l);
    }

    if(l.xnor){
<<<<<<< HEAD
<<<<<<< HEAD
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
        swap_binary(&l);
        binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input = l.binary_input_gpu;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
<<<<<<< HEAD
<<<<<<< HEAD
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
                state.input,
                l.filterDesc,
                l.filters_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
<<<<<<< HEAD
<<<<<<< HEAD
        im2col_ongpu(net.input_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.workspace);
        float * a = l.weights_gpu;
        float * b = net.workspace;
=======
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        float * a = l.filters_gpu;
        float * b = state.workspace;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        float * a = l.filters_gpu;
        float * b = state.workspace;
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
#endif

    if (l.batch_normalize) {
<<<<<<< HEAD
<<<<<<< HEAD
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
=======
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

<<<<<<< HEAD
<<<<<<< HEAD
__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.);
    int h_offset = -(size/2.);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    }
    float *original_input = state.input;

    if(l.xnor) state.input = l.binary_input_gpu;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
<<<<<<< HEAD
<<<<<<< HEAD
            net.input_gpu,
=======
            state.input,
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
            state.input,
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
<<<<<<< HEAD
<<<<<<< HEAD
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
            state.workspace,
            l.workspace_size,
            &one,
            l.dfilterDesc,
            l.filter_updates_gpu);

    if(state.delta){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.filterDesc,
                l.filters_gpu,
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
<<<<<<< HEAD
<<<<<<< HEAD
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
<<<<<<< HEAD
<<<<<<< HEAD
        float * b = net.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(net.input_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(net.delta_gpu){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = net.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        float * b = state.workspace;
        float * c = l.filter_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.filters_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
<<<<<<< HEAD
<<<<<<< HEAD
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
=======
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
<<<<<<< HEAD
<<<<<<< HEAD
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

void push_convolutional_layer(convolutional_layer layer)
{
<<<<<<< HEAD
<<<<<<< HEAD
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
=======
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
<<<<<<< HEAD
<<<<<<< HEAD
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_ongpu(n, B1, m, 1);
    scal_ongpu(n, B2, v, 1);
    axpy_ongpu(n, -decay*batch, w, 1, d, 1);

    axpy_ongpu(n, (1-B1), d, 1, m, 1);
    mul_ongpu(n, d, 1, d, 1);
    axpy_ongpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate/batch, eps, t);
    fill_ongpu(n, 0, d, 1);
}

void update_convolutional_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
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
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);

    axpy_ongpu(size, -decay*batch, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}


