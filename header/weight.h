#ifndef WEIGHT_H
#define WEIGHT_H

#include <CL/cl.h>

cl_float* getWeight(int,int);
char* getPath(int,int);
void generateWeight(cl_float*,int);
int readWeightFile(const char*,cl_float*,int);
int writeWeightFile(const char*,cl_float*,int);
void printWeight(cl_float*,int,int);

#endif