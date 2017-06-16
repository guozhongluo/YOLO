#ifndef LIST_H
#define LIST_H
<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

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
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

<<<<<<< HEAD
<<<<<<< HEAD

=======
void **list_to_array(list *l);

void free_list(list *l);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
void **list_to_array(list *l);

void free_list(list *l);
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
void free_list_contents(list *l);

#endif
