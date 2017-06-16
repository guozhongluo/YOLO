#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
void backward_avgpool_layer_gpu(avgpool_layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_avgpool_layer(const avgpool_layer l, network_state state);
void backward_avgpool_layer(const avgpool_layer l, network_state state);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network_state state);
void backward_avgpool_layer_gpu(avgpool_layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif

