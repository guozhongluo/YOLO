#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
<<<<<<< HEAD
=======
char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

#endif
