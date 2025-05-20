cmake_minimum_required(VERSION 3.10)
project(SNN)
set(C_STANDARD 11)

#add OpenCL and setup
find_package(OpenCL REQUIRED)
add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

#change output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

#kernal conversion (only on linux)
find_program(XXD xxd)
find_program(SED sed)
set(KERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernal)
add_custom_command(
    OUTPUT kernal/forwordPropagation_kernal.h
    COMMAND ${XXD} -i ${KERNAL_DIR}/forwordPropagation.cl |
    ${SED} -e "s/unsigned char .*forwordPropagation_cl/unsigned char forwordProp_kernal_cl/g"
    -e "s/unsigned int .*forwordPropagation_cl_len/unsigned int forwordProp_kernal_cl_len/g" >
    ${KERNAL_DIR}/forwordPropagation_kernal.h
    DEPENDS ${KERNAL_DIR}/forwordPropagation.cl
)

# basic add files, headers and librarys
add_executable(${PROJECT_NAME} src/main.c src/forwordPropagation.c)
target_include_directories(${PROJECT_NAME} PRIVATE kernal header)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)