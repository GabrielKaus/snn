#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include "forwordPropagation.h"
#include "forwordPropagation_kernal.h"

forwordProp_cl* createForwordProp_cl(int n_layers, int neuronsPerLayer, cl_float* waight, cl_int* CL_err_ppr, int* err){
    //create the struct
    forwordProp_cl *cl = (forwordProp_cl*)malloc(sizeof(forwordProp_cl));
    if(cl == NULL){ *err = 1; return NULL;}
    *cl = (forwordProp_cl){n_layers,neuronsPerLayer,LEN,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    //create base
    if(createForwordProp_cl_createBase(cl, CL_err_ppr, err))
        return NULL;
    //create buffers
    if(createForwordProp_cl_createBuffers(cl,waight,CL_err_ppr,err))
        return NULL;
    //create kernal (GPU function)
    if(createForwordProp_cl_createKeranals(cl, CL_err_ppr, err))
        return NULL;
    //ensure that nothing is stuck int the queue
    clFinish(cl->queue);
    *err = 0;
    *CL_err_ppr = CL_SUCCESS;
    return cl;
}

int createForwordProp_cl_createBase(forwordProp_cl* cl, cl_int* CL_err_ppr, int* err){
    cl_int CL_err = CL_SUCCESS;
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    const char* kernel_src = (const char*)forwordProp_kernal_cl;
    const size_t kernel_len = forwordProp_kernal_cl_len;
    //get a single GPU
    CL_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &cl->device, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 2, err, cl)) return 1;
    //create a context to transfer information
    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 3, err, cl)) return 1;
    //create queue to order process in GPU
    cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, props, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 4, err, cl)) return 1;
    //create program
    cl->program = clCreateProgramWithSource(cl->context, 1, &kernel_src, &kernel_len, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 5, err, cl)) return 1;
    //build program
    CL_err = clBuildProgram(cl->program, 1, &cl->device, NULL, NULL, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 6, err, cl))
        return 1;
    return 0;
}

int createForwordProp_cl_createBuffers(forwordProp_cl* cl, cl_float* waight, cl_int* CL_err_ppr, int* err){
    cl_int CL_err = CL_SUCCESS;
    int Vsize,Ssize,Wsize;
    cl_float* zeros;
    //  CREATE BUFFERS
    Vsize = cl->n_layers*cl->neuronsPerLayer;
    cl->V = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Vsize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 7, err, cl)) return 1;
    Ssize = cl->neuronsPerLayer*LEN;
    cl->S = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_uchar)*Ssize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 8, err, cl)) return 1;
    Wsize = cl->n_layers*cl->neuronsPerLayer*cl->neuronsPerLayer;
    cl->W = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Wsize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 10, err, cl)) return 1;
    cl->mult = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Ssize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 11, err, cl)) return 1;
    // SET INITIAL VALUES
    // voltage start with 0
    zeros = (cl_float*)calloc(sizeof(cl_float), Vsize);
    if(zeros == NULL){ *err = 15; return 1;}
    CL_err = clEnqueueWriteBuffer(cl->queue, cl->V, CL_FALSE, 0, sizeof(cl_float)*Vsize, zeros, 0, NULL, NULL);
    free(zeros);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 12, err, cl)) return 1;
    // waight
    CL_err = clEnqueueWriteBuffer(cl->queue, cl->W, CL_FALSE, 0, sizeof(cl_float)*Wsize, waight, 0, NULL, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 13, err, cl))
        return 1;
    return 0;
}

int createForwordProp_cl_createKeranals(forwordProp_cl* cl, cl_int* CL_err_ppr, int* err){
    cl_int CL_err = CL_SUCCESS;
    // ffp kernal
    cl->ffp = clCreateKernel(cl->program, "ffp", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 14, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->ffp, 0, sizeof(cl_mem), &cl->V);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 15, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 16, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 2, sizeof(cl_mem), &cl->mult);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    // dotProd kernal
    cl->dotProd = clCreateKernel(cl->program, "dotProd", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 19, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->dotProd, 0, sizeof(cl_mem), &cl->W);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 2, sizeof(cl_mem), &cl->mult);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 3, sizeof(int), &cl->neuronsPerLayer);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 4, sizeof(int), &cl->len);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 5, sizeof(int), &cl->iteration);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 18, err, cl)) return 1;
    return 0;
}

int err_createForwordProp_cl(cl_int CL_err, cl_int* CL_err_ppr, int err_type, int* err, forwordProp_cl* cl){
    if(CL_err == CL_SUCCESS)
        return 0;
    *CL_err_ppr = CL_err;
    *err = err_type;
    if(cl->device != NULL) clReleaseDevice(cl->device);
    if(cl->context != NULL) clReleaseContext(cl->context);
    if(cl->queue != NULL) clReleaseCommandQueue(cl->queue);
    if(cl->program != NULL) clReleaseProgram(cl->program);
    if(cl->ffp != NULL) clReleaseKernel(cl->ffp);
    if(cl->dotProd != NULL) clReleaseKernel(cl->dotProd);
    if(cl->V != NULL) clReleaseMemObject(cl->V);
    if(cl->S != NULL) clReleaseMemObject(cl->S);
    if(cl->W != NULL) clReleaseMemObject(cl->W);
    if(cl->mult != NULL) clReleaseMemObject(cl->mult);
    free(cl);
    return 1;
}

void print_createForwordProp_cl_error(cl_int CL_err, int err){
    if(err == 0)
        return;
    if(err != 1 && err != 15)
        printf("opencl error: %d\n", CL_err);
    printf("error on create forwordProp: %d\n", err);
}

void releaseForwordProp_cl(forwordProp_cl* cl){
    clReleaseDevice(cl->device);
    clReleaseContext(cl->context);
    clReleaseCommandQueue(cl->queue);
    clReleaseProgram(cl->program);
    clReleaseKernel(cl->ffp);
    clReleaseKernel(cl->dotProd);
    clReleaseMemObject(cl->V);
    clReleaseMemObject(cl->S);
    clReleaseMemObject(cl->W);
    clReleaseMemObject(cl->mult);
    free(cl);
}

int runForwordProp_cl(forwordProp_cl* cl, cl_uchar* array, cl_float* out, cl_int* CL_err){
    const size_t dim[] = {cl->neuronsPerLayer,cl->len,1};
    int Wsize = cl->n_layers*cl->neuronsPerLayer*cl->neuronsPerLayer;
    int Ssize = cl->neuronsPerLayer*LEN;
    printf("\n\ntest %d\n", Ssize);
    for(int i=0; i<Ssize; i++){
        printf("S[%d]=%d ", i, array[i]);
    }
    printf("\n\n");
    //write
    *CL_err = clEnqueueWriteBuffer(cl->queue, cl->S, CL_TRUE, 0, sizeof(cl_uchar)*Ssize, array, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 1;
    //execute
    *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->dotProd, 2, NULL, dim, NULL, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 2;
    //read
    *CL_err = clEnqueueReadBuffer(cl->queue, cl->mult, CL_TRUE, 0, sizeof(cl_float)*Ssize, out, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 3;
    clFinish(cl->queue);
    return 0;
}