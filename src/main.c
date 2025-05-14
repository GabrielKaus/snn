#include <stdio.h>
#include <CL/cl.h>
#include "forwordPropagation_kernal.h"

void printFloatList(float*,int);

int main()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    if (CL_err == CL_SUCCESS){
        printf("%u platform(s) found\n", numPlatforms);
    }else{
        printf("clGetPlatformIDs(%i)\n", CL_err);
        return 1;
    }

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;

    int size = 500;
    float data[size];
    for(int i=0; i<size; i++){
        data[i] = (float)i / 5;
    };
    printFloatList(data, size);

    CL_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    if (CL_err == CL_SUCCESS){
        printf("Get GPU\n");
    }else{
        printf("Dont get GPU, Erro(%i)\n", CL_err);
        return 1;
    }
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, device, (cl_command_queue_properties)0, NULL);
    
    char *source = forwordProp_kernal_cl;
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &CL_err);
    if(CL_err != CL_SUCCESS){
        printf("erro on create program with source");
        return 1;
    }
    CL_err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(CL_err != CL_SUCCESS){
        printf("erro on build program (%i)", CL_err);
        return 1;
    }
    kernel = clCreateKernel(program, "calcSquare", &CL_err);
    if(CL_err != CL_SUCCESS){
        printf("erro on create kernel");
        return 1;
    }
    
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*size, NULL, &CL_err);
    if(CL_err != CL_SUCCESS){
        printf("erro on create buffer (%i)", CL_err);
        return 1;
    }
    clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, sizeof(cl_float)*size, &data, 0, NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    size_t dim[] = {size,0,0};
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, dim, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, sizeof(cl_float)*size, &data, 0, NULL, NULL);

    clFinish(queue);
    printFloatList(data, size);
    return 0;
}

void printFloatList(float* data, int size){
    for(int i=0; i<size; i++){
        printf("%f ", data[i]);
    }
    printf("\n\n");
}