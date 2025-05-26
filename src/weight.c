#include <stdio.h>
#include <CL/cl.h>
#include "weight.h"

cl_float* getWeight(int nl, int npl){
    int test;
    char path[30];
    int Wsize = nl*npl*npl;
    cl_float* weight = (cl_float*)malloc(sizeof(cl_float)*Wsize);
    if(weight == NULL) return NULL;
    sprintf(path, "weight/w_%d_%d.bin", nl, npl);
    if(!readWeightFile(path, weight, Wsize))
        return weight;
    generateWeight(weight,Wsize);
    writeWeightFile(path, weight, Wsize);
    return weight;
}

void generateWeight(cl_float* weight, int size){
    for(int i=0; i<size; i++)
        weight[i] = (cl_float)rand()/RAND_MAX;
}

int readWeightFile(const char* path, cl_float* weight, int size){
    FILE* file = fopen(path, "rb");
    if(file == NULL) return 1;
    fread(weight, sizeof(cl_float), size, file);
    return 0;
}

int writeWeightFile(const char* path, cl_float* weight, int size){
    FILE* file = fopen(path, "wb");
    if(file == NULL) return 1;
    fwrite(weight, sizeof(cl_float), size, file);
    fclose(file);
    return 0;
}

void printWeight(cl_float* weight, int nl, int npl){
    for(int i=0; i<nl; i++){
        for(int j=0; j<npl; j++){
            for(int k=0; k<npl; k++)
                printf("%f ", weight[i*npl*npl + j*npl + k]);
            printf("\n");
        }
        printf("\n");
    }
}