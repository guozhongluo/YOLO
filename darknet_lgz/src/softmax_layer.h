#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_softmax_layer(const softmax_layer l, network_state state);
void backward_softmax_layer(const softmax_layer l, network_state state);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network_state state);
void backward_softmax_layer_gpu(const softmax_layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif
