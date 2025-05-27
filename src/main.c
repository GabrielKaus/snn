#include <stdio.h>
#include <stdlib.h>
#include "forwordPropagation.h"
#include "weight.h"

void printFloatList(float*,int,int);
void printClUchar(cl_uchar*,int,int);

int main()
{
    cl_int CL_err = CL_SUCCESS;
    int err = 0;
    forwordProp_cl* cl;
    int nl = 2;
    int npl = 4;
    cl_float* weight = getWeight(nl, npl);
    printWeight(weight, nl, npl);
    cl_uchar data[] = {5, 10};
    cl = createForwordProp_cl(nl, npl, weight, &CL_err, &err);
    if(cl == NULL){
        print_createForwordProp_cl_error(CL_err, err);
        return 1;
    }
    err = runForwordProp_cl(cl, data, &CL_err);
    if(err != 0){
        printf("opencl error: %d\nerror on run forwordProp: %d\n", CL_err, err);
        return 1;
    }
    printClUchar(data,npl/2,1);
    releaseForwordProp_cl(cl);
    return 0;
}

void printFloatList(float* data, int size, int len){
    for(int i=0; i<size; i++){
        for(int j=0; j<len; j++){
            printf("%f ", data[i*len + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printClUchar(cl_uchar* data, int size, int len){
    for(int i=0; i<size; i++){
        for(int j=0; j<len; j++){
            printf("%d ", data[i*len + j]);
        }
        printf("\n");
    }
    printf("\n");
}