#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer *layer, int h, int w);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_normalization_layer(const layer layer, network net);
void backward_normalization_layer(const layer layer, network net);
void visualize_normalization_layer(layer layer, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net);
void backward_normalization_layer_gpu(const layer layer, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_normalization_layer(const layer layer, network_state state);
void backward_normalization_layer(const layer layer, network_state state);
void visualize_normalization_layer(layer layer, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network_state state);
void backward_normalization_layer_gpu(const layer layer, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif
