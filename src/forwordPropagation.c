#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include "forwordPropagation.h"
#include "forwordPropagation_kernal.h"

forwordProp_cl* createForwordProp_cl(int size, cl_int* CL_err_ppr, int* err){
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_int CL_err = CL_SUCCESS;
    const char* kernel_src = (const char*)forwordProp_kernal_cl;
    const size_t kernel_len = forwordProp_kernal_cl_len;
    //create the struct
    forwordProp_cl *cl = (forwordProp_cl*)malloc(sizeof(forwordProp_cl));
    if(cl == NULL){
        *err = 1;
        return NULL;
    }
    cl->device = NULL;
    cl->context = NULL;
    cl->queue = NULL;
    cl->program = NULL;
    cl->kernel = NULL;
    cl->buffer = NULL;
    //get a single GPU
    CL_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &cl->device, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 2, err, cl))
        return NULL;
    //create a context to transfer information
    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 3, err, cl))
        return NULL;
    //create queue to order process in GPU
    cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, props, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 4, err, cl))
        return NULL;
    //create program
    cl->program = clCreateProgramWithSource(cl->context, 1, &kernel_src, &kernel_len, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 5, err, cl))
        return NULL;
    //build program
    CL_err = clBuildProgram(cl->program, 1, &cl->device, NULL, NULL, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 6, err, cl))
        return NULL;
    //create kernal (GPU function)
    cl->kernel = clCreateKernel(cl->program, "calcSquare", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 7, err, cl))
        return NULL;
    //create global buffer (memory on the GPU)
    cl->buffer = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float)*size, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 8, err, cl))
        return NULL;
    //set the buffer as the input of the kernal
    CL_err = clSetKernelArg(cl->kernel, 0, sizeof(cl_mem), &cl->buffer);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 9, err, cl))
        return NULL;
    //ensure that nothing is stuck int the queue
    clFinish(cl->queue);
    *err = 0;
    *CL_err_ppr = CL_SUCCESS;
    return cl;
}

int err_createForwordProp_cl(cl_int CL_err, cl_int* CL_err_ppr, int err_type, int* err, forwordProp_cl* cl){
    if(CL_err == CL_SUCCESS)
        return 0;
    *CL_err_ppr = CL_err;
    *err = err_type;
    if(cl->device != NULL)
        clReleaseDevice(cl->device);
    if(cl->context != NULL)
        clReleaseContext(cl->context);
    if(cl->queue != NULL)
        clReleaseCommandQueue(cl->queue);
    if(cl->program != NULL)
        clReleaseProgram(cl->program);
    if(cl->kernel != NULL)
        clReleaseKernel(cl->kernel);
    if(cl->buffer != NULL)
        clReleaseMemObject(cl->buffer);
    free(cl);
    return 1;
}

void print_createForwordProp_cl_error(cl_int CL_err, int err){
    if(err == 0)
        return;
    if(err != 9)
        printf("opencl error: %d\n", CL_err);
    printf("error on create forwordProp: %d\n", err);
}

int runForwordProp_cl(forwordProp_cl* cl, float* array, int size, cl_int* CL_err){
    const size_t dim[] = {size,0,0};
    //write
    *CL_err = clEnqueueWriteBuffer(cl->queue, cl->buffer, CL_FALSE, 0, sizeof(cl_float)*size, array, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 1;
    //execute
    *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->kernel, 1, NULL, dim, NULL, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 2;
    //read
    *CL_err = clEnqueueReadBuffer(cl->queue, cl->buffer, CL_FALSE, 0, sizeof(cl_float)*size, array, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 3;
    clFinish(cl->queue);
    return 0;
}

void releaseForwordProp_cl(forwordProp_cl* cl){
    clReleaseDevice(cl->device);
    clReleaseContext(cl->context);
    clReleaseCommandQueue(cl->queue);
    clReleaseProgram(cl->program);
    clReleaseKernel(cl->kernel);
    clReleaseMemObject(cl->buffer);
    free(cl);
}