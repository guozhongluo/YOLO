#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer crop_layer;

image get_crop_image(crop_layer l);
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_crop_layer(const crop_layer l, network net);
void resize_crop_layer(layer *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_crop_layer(const crop_layer l, network_state state);
void resize_crop_layer(layer *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif

