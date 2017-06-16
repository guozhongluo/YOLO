
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);

<<<<<<< HEAD
<<<<<<< HEAD
void forward_crnn_layer(layer l, network net);
void backward_crnn_layer(layer l, network net);
void update_crnn_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_crnn_layer_gpu(layer l, network net);
void backward_crnn_layer_gpu(layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_crnn_layer(layer l, network_state state);
void backward_crnn_layer(layer l, network_state state);
void update_crnn_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_crnn_layer_gpu(layer l, network_state state);
void backward_crnn_layer_gpu(layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void update_crnn_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
void push_crnn_layer(layer l);
void pull_crnn_layer(layer l);
#endif

#endif

