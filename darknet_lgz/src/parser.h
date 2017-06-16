#ifndef PARSER_H
#define PARSER_H
<<<<<<< HEAD
<<<<<<< HEAD
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
=======
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592
#include "network.h"

network parse_network_cfg(char *filename);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);
void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);
<<<<<<< HEAD
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445
=======
>>>>>>> 07267f401b3d9c82c5f695f932c9f504d2b6a592

#endif
