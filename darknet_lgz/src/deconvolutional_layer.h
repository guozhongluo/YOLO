#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
<<<<<<< HEAD
#ifdef GPU
void forward_deconvolutional_layer_gpu(layer l, network net);
void backward_deconvolutional_layer_gpu(layer l, network net);
void update_deconvolutional_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(layer l);
void pull_deconvolutional_layer(layer l);
#endif

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(const layer l, network net);
void update_deconvolutional_layer(layer l, int batch, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
typedef layer deconvolutional_layer;

#ifdef GPU
void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer_gpu(deconvolutional_layer layer, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(deconvolutional_layer layer);
void pull_deconvolutional_layer(deconvolutional_layer layer);
#endif

deconvolutional_layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void resize_deconvolutional_layer(deconvolutional_layer *layer, int h, int w);
void forward_deconvolutional_layer(const deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer(deconvolutional_layer layer, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(deconvolutional_layer layer, network_state state);

image get_deconvolutional_image(deconvolutional_layer layer);
image get_deconvolutional_delta(deconvolutional_layer layer);
image get_deconvolutional_filter(deconvolutional_layer layer, int i);

int deconvolutional_out_height(deconvolutional_layer layer);
int deconvolutional_out_width(deconvolutional_layer layer);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

#endif

