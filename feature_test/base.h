/**
 * Macros, Enums and Inline functions
**/

#ifndef SPMM_UTILS_BASE_H
#define SPMM_UTILS_BASE_H

#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#include <omp.h>
#include <pthread.h>

//MACROS
#define DEFAULT_THREADS_FOR_DENSE_MAT 6

#define MM_DEFAULT_DENSE_MAT_COL 1024
#define MM_DEFAULT_DENSE_COL_BLOCK 8

#define DEFAULT_ROW_BLOCK_SIZE 32
#define DEFAULT_COL_BLOCK_SIZE 32
#define WARP_PER_BLOCK 16
#define WARP_SIZE 32

#define TC_BLK_H 16
#define TC_BLK_W 16

#define HALF_MAX 65520.f
#define HALF_MIN 0.00000006f

//memory controller
#define MEMORY_BOUND 228 //in GB
#define A_MEMORY_BOUND 120
#define WORKSPACE_BOUND 36

void spmm_usage(char* prog);

void parameter_parser(int argc, char* argv[], 
                      int64_t &dense_col_dim,
                      std::string &input_file);

#endif