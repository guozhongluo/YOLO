#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"
#include "list.h"

#define TWO_PI 6.2831853071795864769252866

double what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "list.h"

#define SECRET_NUM -1234

void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
char *basecfg(char *cfgfile);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
<<<<<<< HEAD
<<<<<<< HEAD
void find_replace(char *str, char *orig, char *rep, char *output);
=======
char *find_replace(char *str, char *orig, char *rep);
void error(const char *s);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
char *find_replace(char *str, char *orig, char *rep);
void error(const char *s);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
<<<<<<< HEAD
<<<<<<< HEAD
=======
void top_k(float *a, int n, int k, int *index);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
void top_k(float *a, int n, int k, int *index);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
<<<<<<< HEAD
<<<<<<< HEAD
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float sum_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
void print_statistics(float *a, int n);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void normalize_array(float *a, int n);
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
int max_index(float *a, int n);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float mse_array(float *a, int n);
float rand_normal();
size_t rand_size_t();
float rand_uniform(float min, float max);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

#endif

