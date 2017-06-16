<<<<<<< HEAD
#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H
=======
#ifndef REGION_LAYER_H
#define REGION_LAYER_H
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

#include "layer.h"
#include "network.h"

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
<<<<<<< HEAD
void forward_detection_layer(const detection_layer l, network net);
void backward_detection_layer(const detection_layer l, network net);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network net);
void backward_detection_layer_gpu(detection_layer l, network net);
=======
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
#endif

#endif
