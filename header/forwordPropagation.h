#ifndef FORWORD_PROPAGATION_H
#define FORWORD_PROPAGATION_H

#include <CL/cl.h>

typedef struct forwordProp_cl{
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;
} forwordProp_cl;

forwordProp_cl* createForwordProp_cl(int,cl_int*,int*);
int err_createForwordProp_cl(cl_int,cl_int*,int,int*,forwordProp_cl*);
void print_createForwordProp_cl_error(cl_int,int);
int runForwordProp_cl(forwordProp_cl*,float*,int,cl_int*);
void releaseForwordProp_cl(forwordProp_cl*);

#endif