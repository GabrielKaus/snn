#include <stdio.h>
#include <stdlib.h>
#include "forwordPropagation.h"

void printFloatList(float*,int,int);
void printClUchar(cl_uchar*,int,int);

int main()
{
    cl_int CL_err = CL_SUCCESS;
    int err = 0;
    forwordProp_cl* cl;
    int nl = 2;
    int npl = 3;
    cl_float weight[] = {
        0.000, 0.100, 0.200,
        0.010, 0.110, 0.210,
        0.020, 0.120, 0.220,
        0.001, 0.101, 0.202,
        0.011, 0.111, 0.211,
        0.021, 0.121, 0.221
    };
    printFloatList(weight, npl, npl);
    cl_uchar S[] = {
        1, 0, 1,
        0, 1, 0,
        0, 1, 1
    };
    printClUchar(S, npl, LEN);
    cl = createForwordProp_cl(nl, npl, weight, &CL_err, &err);
    if(cl == NULL){
        print_createForwordProp_cl_error(CL_err, err);
        return 1;
    }
    err = runForwordProp_cl(cl, S, &CL_err);
    if(err != 0){
        printf("opencl error: %d\nerror on run forwordProp: %d\n", CL_err, err);
        return 1;
    }
    printClUchar(S, npl, LEN);
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