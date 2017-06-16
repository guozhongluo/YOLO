#ifndef BOX_H
#define BOX_H
<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

typedef struct{
    float x, y, w, h;
} box;
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

typedef struct{
    float dx, dy, dw, dh;
} dbox;

<<<<<<< HEAD
<<<<<<< HEAD
box float_to_box(float *f, int stride);
float box_rmse(box a, box b);
dbox diou(box a, box b);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
