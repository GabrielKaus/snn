#ifndef FORWORD_PROPAGATION_H
#define FORWORD_PROPAGATION_H

#include <CL/cl.h>

typedef struct forwordProp_cl{
    const cl_device_id device;
    const cl_context context;
    const cl_command_queue queue;
    const cl_program program;
    const cl_kernel kernel;
    const cl_mem buffer;
} forwordProp_cl;

forwordProp_cl* createForwordProp_cl(int*);
void clearForwordProp_cl(forwordProp_cl*);

#endif