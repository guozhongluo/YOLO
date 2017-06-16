#ifndef PARSER_H
#define PARSER_H
<<<<<<< HEAD
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
=======
#include "network.h"

network parse_network_cfg(char *filename);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);
void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);
>>>>>>> b5b3d7367411302dd6e73c8fe583d6860a786445

#endif
