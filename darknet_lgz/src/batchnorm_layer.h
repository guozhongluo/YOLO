#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_batchnorm_layer(layer l, network net);
void backward_batchnorm_layer(layer l, network net);

#ifdef GPU
void forward_batchnorm_layer_gpu(layer l, network net);
void backward_batchnorm_layer_gpu(layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_batchnorm_layer(layer l, network_state state);
void backward_batchnorm_layer(layer l, network_state state);

#ifdef GPU
void forward_batchnorm_layer_gpu(layer l, network_state state);
void backward_batchnorm_layer_gpu(layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);
#endif

#endif
