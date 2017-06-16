<<<<<<< HEAD
<<<<<<< HEAD
#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H
=======
#ifndef REGION_LAYER_H
#define REGION_LAYER_H
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
#ifndef REGION_LAYER_H
#define REGION_LAYER_H
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

#include "layer.h"
#include "network.h"

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_detection_layer(const detection_layer l, network net);
void backward_detection_layer(const detection_layer l, network net);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network net);
void backward_detection_layer_gpu(detection_layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif
