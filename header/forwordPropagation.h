#ifndef FORWORD_PROPAGATION_H
#define FORWORD_PROPAGATION_H

#include <CL/cl.h>

#define LEN 15
///16

typedef struct forwordProp_cl{
    int n_layers, neuronsPerLayer, len, iteration;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel ffp, dotProd, writeData, readData;
    cl_mem V, S, W, mult, data;
} forwordProp_cl;

forwordProp_cl* createForwordProp_cl(int,int,cl_float*,cl_int*,int*);
int createForwordProp_cl_createBase(forwordProp_cl*,cl_int*,int*);
int createForwordProp_cl_createBuffers(forwordProp_cl*,cl_float*,cl_int*,int*);
int createForwordProp_cl_createKeranals(forwordProp_cl*,cl_int*,int*);
int err_createForwordProp_cl(cl_int,cl_int*,int,int*,forwordProp_cl*);
void print_createForwordProp_cl_error(cl_int,int);
void releaseForwordProp_cl(forwordProp_cl*);
int writeForwordProp_cl(forwordProp_cl*,cl_uchar*,cl_int*);
int readForwordProp_cl(forwordProp_cl*,cl_uchar*,cl_int*);
int runForwordProp_cl(forwordProp_cl*,cl_uchar*,cl_int*);

#endif