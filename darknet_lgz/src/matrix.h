#ifndef MATRIX_H
#define MATRIX_H
<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"

matrix copy_matrix(matrix m);
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
void print_matrix(matrix m);

matrix csv_to_matrix(char *filename);
void matrix_to_csv(matrix m);
matrix hold_out_matrix(matrix *m, int n);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#endif
