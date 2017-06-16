#ifndef BOX_H
#define BOX_H
<<<<<<< HEAD
#include "darknet.h"
=======

typedef struct{
    float x, y, w, h;
} box;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

typedef struct{
    float dx, dy, dw, dh;
} dbox;

<<<<<<< HEAD
box float_to_box(float *f, int stride);
float box_rmse(box a, box b);
dbox diou(box a, box b);
=======
box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
