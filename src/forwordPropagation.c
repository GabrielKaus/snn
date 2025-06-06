#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include "forwordPropagation.h"
#include "forwordPropagation_kernal.h"

forwordProp_cl* createForwordProp_cl(int n_layers, int neuronsPerLayer, cl_float* waight, cl_int* CL_err_ppr, int* err){
    //create the struct
    forwordProp_cl *cl = (forwordProp_cl*)malloc(sizeof(forwordProp_cl));
    if(cl == NULL){ *err = 1; return NULL;}
    *cl = (forwordProp_cl){n_layers,neuronsPerLayer,LEN,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
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
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 100, err, cl)) return 1;
    //create a context to transfer information
    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 101, err, cl)) return 1;
    //create queue to order process in GPU
    cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, props, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 102, err, cl)) return 1;
    //create program
    cl->program = clCreateProgramWithSource(cl->context, 1, &kernel_src, &kernel_len, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 103, err, cl)) return 1;
    //build program
    CL_err = clBuildProgram(cl->program, 1, &cl->device, NULL, NULL, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 104, err, cl))
        return 1;
    return 0;
}

int createForwordProp_cl_createBuffers(forwordProp_cl* cl, cl_float* waight, cl_int* CL_err_ppr, int* err){
    cl_int CL_err = CL_SUCCESS;
    int Vsize,Ssize,Wsize,Dsize;
    cl_float* zeros;
    //  CREATE BUFFERS
    Vsize = cl->n_layers*cl->neuronsPerLayer;
    cl->V = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Vsize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 200, err, cl)) return 1;
    Ssize = cl->neuronsPerLayer*LEN;
    cl->S = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_uchar)*Ssize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 201, err, cl)) return 1;
    Wsize = cl->n_layers*cl->neuronsPerLayer*cl->neuronsPerLayer;
    cl->W = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Wsize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 202, err, cl)) return 1;
    cl->mult = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_float)*Ssize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 203, err, cl)) return 1;
    Dsize = cl->neuronsPerLayer/2;
    cl->data = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(cl_uchar)*Dsize, NULL, &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 204, err, cl)) return 1;
    // SET INITIAL VALUES
    // voltage start with 0
    zeros = (cl_float*)calloc(sizeof(cl_float), Vsize);
    if(zeros == NULL){ *err = 300; return 1;}
    CL_err = clEnqueueWriteBuffer(cl->queue, cl->V, CL_FALSE, 0, sizeof(cl_float)*Vsize, zeros, 0, NULL, NULL);
    free(zeros);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 301, err, cl)) return 1;
    // waight
    CL_err = clEnqueueWriteBuffer(cl->queue, cl->W, CL_FALSE, 0, sizeof(cl_float)*Wsize, waight, 0, NULL, NULL);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 302, err, cl))
        return 1;
    return 0;
}

int createForwordProp_cl_createKeranals(forwordProp_cl* cl, cl_int* CL_err_ppr, int* err){
    cl_int CL_err = CL_SUCCESS;
    // ffp kernal
    cl->ffp = clCreateKernel(cl->program, "ffp", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 400, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->ffp, 0, sizeof(cl_mem), &cl->V);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 401, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 402, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 2, sizeof(cl_mem), &cl->mult);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 403, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 3, sizeof(int), &cl->n_layers);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 404, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 4, sizeof(int), &cl->len);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 405, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->ffp, 5, sizeof(int), &cl->iteration);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 406, err, cl)) return 1;
    // dotProd kernal
    cl->dotProd = clCreateKernel(cl->program, "dotProd", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 500, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->dotProd, 0, sizeof(cl_mem), &cl->W);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 501, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 502, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 2, sizeof(cl_mem), &cl->mult);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 503, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 3, sizeof(int), &cl->neuronsPerLayer);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 504, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 4, sizeof(int), &cl->len);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 505, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->dotProd, 5, sizeof(int), &cl->iteration);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 506, err, cl)) return 1;
    // writeData kernal
    cl->writeData = clCreateKernel(cl->program, "writeData", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 600, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->writeData, 0, sizeof(cl_mem), &cl->data);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 601, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->writeData, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 602, err, cl)) return 1;
    // readData kernal
    cl->readData = clCreateKernel(cl->program, "readData", &CL_err);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 700, err, cl)) return 1;
    //set kernal variables
    CL_err = clSetKernelArg(cl->readData, 0, sizeof(cl_mem), &cl->data);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 701, err, cl)) return 1;
    CL_err = clSetKernelArg(cl->readData, 1, sizeof(cl_mem), &cl->S);
    if(err_createForwordProp_cl(CL_err, CL_err_ppr, 702, err, cl)) return 1;
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
    if(cl->writeData != NULL) clReleaseKernel(cl->writeData);
    if(cl->readData != NULL) clReleaseKernel(cl->readData);
    if(cl->V != NULL) clReleaseMemObject(cl->V);
    if(cl->S != NULL) clReleaseMemObject(cl->S);
    if(cl->W != NULL) clReleaseMemObject(cl->W);
    if(cl->mult != NULL) clReleaseMemObject(cl->mult);
    if(cl->data != NULL) clReleaseMemObject(cl->data);
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
    clReleaseKernel(cl->writeData);
    clReleaseKernel(cl->readData);
    clReleaseMemObject(cl->V);
    clReleaseMemObject(cl->S);
    clReleaseMemObject(cl->W);
    clReleaseMemObject(cl->mult);
    clReleaseMemObject(cl->data);
    free(cl);
}

int writeForwordProp_cl(forwordProp_cl* cl, cl_uchar* data, cl_int* CL_err){
    int size = cl->neuronsPerLayer/2;
    const size_t dim[] = {size,1,1};
    *CL_err = clEnqueueWriteBuffer(cl->queue, cl->data, CL_TRUE, 0, sizeof(cl_uchar)*size, data, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 1;
    *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->writeData, 1, NULL, dim, NULL, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 2;
    return 0;
}

int readForwordProp_cl(forwordProp_cl* cl, cl_uchar* data, cl_int* CL_err){
    int size = cl->neuronsPerLayer/2;
    const size_t dim[] = {size,1,1};
    *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->readData, 1, NULL, dim, NULL, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 1;
    *CL_err = clEnqueueReadBuffer(cl->queue, cl->data, CL_FALSE, 0, sizeof(cl_uchar)*size, data, 0, NULL, NULL);
    if(*CL_err != CL_SUCCESS)
        return 2;
    return 0;
}

int runForwordProp_cl(forwordProp_cl* cl, cl_uchar* data, cl_int* CL_err){
    int err;
    const size_t dim_dotProd[] = {cl->neuronsPerLayer,cl->len,1};
    const size_t dim_ffp[] = {cl->neuronsPerLayer,1,1};
    int Wsize = cl->n_layers*cl->neuronsPerLayer*cl->neuronsPerLayer;
    int Ssize = cl->neuronsPerLayer*LEN;
    err = writeForwordProp_cl(cl, data, CL_err);
    if(err) return err;
    //execute
    for(cl->iteration=0; cl->iteration<cl->n_layers; cl->iteration++){
        *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->dotProd, 2, NULL, dim_dotProd, NULL, 0, NULL, NULL);
        if(*CL_err != CL_SUCCESS)
            return 2;
        *CL_err = clEnqueueNDRangeKernel(cl->queue, cl->ffp, 1, NULL, dim_ffp, NULL, 0, NULL, NULL);
        if(*CL_err != CL_SUCCESS)
            return 3;
    }
    err = readForwordProp_cl(cl, data, CL_err);
    if(err) return err+4;
    clFinish(cl->queue);
    return 0;
}