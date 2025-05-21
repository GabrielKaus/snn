#include <stdio.h>
#include "forwordPropagation.h"

void printFloatList(float*,int);

int main()
{
    cl_int CL_err = CL_SUCCESS;
    int err = 0;
    forwordProp_cl* cl;
    int size = 500;
    float data[size];
    //create array
    for(int i=0; i<size; i++){
        data[i] = (float)i / 5;
    };
    printFloatList(data, size);
    cl = createForwordProp_cl(size, 100, &CL_err, &err);
    if(cl == NULL){
        print_createForwordProp_cl_error(CL_err, err);
        return 1;
    }
    err = runForwordProp_cl(cl, data, size, &CL_err);
    if(err != 0){
        printf("opencl error: %d\nerror on run forwordProp: %d\n", CL_err, err);
        return 1;
    }
    printFloatList(data, size);
    releaseForwordProp_cl(cl);
    return 0;
}

void printFloatList(float* data, int size){
    for(int i=0; i<size; i++){
        printf("%f ", data[i]);
    }
    printf("\n\n");
}