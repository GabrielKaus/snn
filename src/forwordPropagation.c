#include <stdio.h>
#include <CL/cl.h>
#include "forwordPropagation.h"
#include "forwordPropagation_kernal.h"

forwordProp_cl* createForwordProp_cl(int* err){
    cl_int CL_err = CL_SUCCESS;
    cl_device_id device;
    *err = 0;
    //get a single GPU
    CL_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(CL_err != CL_SUCCESS){
        printf("opencl error: %d\n", CL_err);
        *err = 1;
        return NULL;
    }
}