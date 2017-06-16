#ifndef LIST_H
#define LIST_H
<<<<<<< HEAD
#include "darknet.h"
=======

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

<<<<<<< HEAD

=======
void **list_to_array(list *l);

void free_list(list *l);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
void free_list_contents(list *l);

#endif
