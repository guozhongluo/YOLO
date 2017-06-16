#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
<<<<<<< HEAD
<<<<<<< HEAD
void forward_shortcut_layer(const layer l, network net);
void backward_shortcut_layer(const layer l, network net);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void forward_shortcut_layer(const layer l, network_state state);
void backward_shortcut_layer(const layer l, network_state state);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network_state state);
void backward_shortcut_layer_gpu(const layer l, network_state state);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#endif

#endif
