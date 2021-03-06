#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
<<<<<<< HEAD
<<<<<<< HEAD
=======
#include "image.h"
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
#include "image.h"
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
<<<<<<< HEAD
<<<<<<< HEAD
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"
}

<<<<<<< HEAD
<<<<<<< HEAD
void forward_network_gpu(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
<<<<<<< HEAD
<<<<<<< HEAD
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(net);
    calc_network_cost(net);
}

void backward_network_gpu(network net)
{
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        if(l.type == CONVOLUTIONAL){
            forward_convolutional_layer_gpu(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            forward_deconvolutional_layer_gpu(l, state);
        } else if(l.type == ACTIVE){
            forward_activation_layer_gpu(l, state);
        } else if(l.type == LOCAL){
            forward_local_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            forward_detection_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            forward_connected_layer_gpu(l, state);
        } else if(l.type == RNN){
            forward_rnn_layer_gpu(l, state);
        } else if(l.type == GRU){
            forward_gru_layer_gpu(l, state);
        } else if(l.type == CRNN){
            forward_crnn_layer_gpu(l, state);
        } else if(l.type == CROP){
            forward_crop_layer_gpu(l, state);
        } else if(l.type == COST){
            forward_cost_layer_gpu(l, state);
        } else if(l.type == SOFTMAX){
            forward_softmax_layer_gpu(l, state);
        } else if(l.type == NORMALIZATION){
            forward_normalization_layer_gpu(l, state);
        } else if(l.type == BATCHNORM){
            forward_batchnorm_layer_gpu(l, state);
        } else if(l.type == MAXPOOL){
            forward_maxpool_layer_gpu(l, state);
        } else if(l.type == AVGPOOL){
            forward_avgpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            forward_dropout_layer_gpu(l, state);
        } else if(l.type == ROUTE){
            forward_route_layer_gpu(l, net);
        } else if(l.type == SHORTCUT){
            forward_shortcut_layer_gpu(l, state);
        }
        state.input = l.output_gpu;
    }
}

void backward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        if(l.type == CONVOLUTIONAL){
            backward_convolutional_layer_gpu(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            backward_deconvolutional_layer_gpu(l, state);
        } else if(l.type == ACTIVE){
            backward_activation_layer_gpu(l, state);
        } else if(l.type == LOCAL){
            backward_local_layer_gpu(l, state);
        } else if(l.type == MAXPOOL){
            if(i != 0) backward_maxpool_layer_gpu(l, state);
        } else if(l.type == AVGPOOL){
            if(i != 0) backward_avgpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            backward_dropout_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            backward_detection_layer_gpu(l, state);
        } else if(l.type == NORMALIZATION){
            backward_normalization_layer_gpu(l, state);
        } else if(l.type == BATCHNORM){
            backward_batchnorm_layer_gpu(l, state);
        } else if(l.type == SOFTMAX){
            if(i != 0) backward_softmax_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            backward_connected_layer_gpu(l, state);
        } else if(l.type == RNN){
            backward_rnn_layer_gpu(l, state);
        } else if(l.type == GRU){
            backward_gru_layer_gpu(l, state);
        } else if(l.type == CRNN){
            backward_crnn_layer_gpu(l, state);
        } else if(l.type == COST){
            backward_cost_layer_gpu(l, state);
        } else if(l.type == ROUTE){
            backward_route_layer_gpu(l, net);
        } else if(l.type == SHORTCUT){
            backward_shortcut_layer_gpu(l, state);
        }
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    }
}

void update_network_gpu(network net)
{
<<<<<<< HEAD
<<<<<<< HEAD
    cuda_set_device(net.gpu_index);
=======
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
<<<<<<< HEAD
<<<<<<< HEAD
        l.t = get_current_batch(net);
        if(l.update_gpu){
            l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == DECONVOLUTIONAL){
            update_deconvolutional_layer_gpu(l, rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == GRU){
            update_gru_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == RNN){
            update_rnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == CRNN){
            update_crnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == LOCAL){
            update_local_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
        }
    }
}

<<<<<<< HEAD
<<<<<<< HEAD
void harmless_update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_ongpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_ongpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_ongpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

float train_network_datum_gpu(network net)
{
    *net.seen += net.batch;

    int x_size = net.inputs*net.batch;
    int y_size = net.truths*net.batch;
    cuda_push_array(net.input_gpu, net.input, x_size);
    cuda_push_array(net.truth_gpu, net.truth, y_size);

    net.train = 1;
    forward_network_gpu(net);
    backward_network_gpu(net);

    float error = *net.cost;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
float train_network_datum_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].type == DETECTION) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
    float error = get_network_cost(net);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}

<<<<<<< HEAD
<<<<<<< HEAD
typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.n*l.size*l.size*l.c, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.n*l.size*l.size*l.c);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
    }
}
void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.n*l.size*l.size*l.c);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}
*/

/*
void sync_layer(network *nets, int n, int j)
{
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
}
*/

void sync_layer(network *nets, int n, int j)
{
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

float *network_predict_gpu(network net, float *input)
{
<<<<<<< HEAD
<<<<<<< HEAD
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, input, net.inputs*net.batch);
    net.truth = 0;
    net.train = 0;
    forward_network_gpu(net);
    return net.output;
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
}

