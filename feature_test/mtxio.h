/* 
*   Matrix Market I/O library in C++
*   Only support Pattern/Integer/Real
*   @property: nnz value in Pattern is set to 1
*   @property: only include the read function for mtx matrices
*   @output: matrix in coo format (two version for symmetric matrix)
*/

#ifndef MTX_IO_PTHREAD_H
#define MTX_IO_PTHREAD_H

#pragma once

#include "./base.h"
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <time.h>

//MTX MAT TYPE
enum MM_MTYPE_CODE {MM_SYMMETRIC, MM_GENERAL};

//MTX DATA TYPE
enum MM_DTYPE_CODE {MM_INTEGER, MM_FLOAT, MM_PATTERN};

//memory compress flag for extremely large symmetric mats
enum MM_SYM_COMPRESS_FLAG {MM_SYM_UNCOMPRESS, MM_SYM_COMPRESS};

enum MM_MTX_GENERATE_METHOD {MM_DENSE, MM_SPARSE_RANDOM, MM_SPARSE_FROM_FILE};

/********************* MM_typecode query fucntions ***************************/

#define mm_is_real(typecode)		((typecode) == MM_FLOAT)
#define mm_is_integer(typecode) ((typecode) == MM_INTEGER)

#define mm_is_symmetric(typecode)((typecode) == MM_SYMMETRIC)
#define mm_is_general(typecode)	((typecode) == MM_GENERAL)
#define mm_is_pattern(typecode)	((typecode) == MM_PATTERN)

/********************* Matrix Market IO error codes 
 ***************************/

#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_UNSUPPORTED_TYPE     14
#define MM_LINE_TOO_LONG		16

/* I/O operations */
void mm_read_banner(std::ifstream &mtx_file, 
                    enum MM_MTYPE_CODE &mtx_mat_type, 
					enum MM_DTYPE_CODE &mtx_data_type,
                    int &mat_support_flag);

void mm_read_mtx_crd_size(std::ifstream &mtx_file, 
                          int64_t &M, int64_t &N, int64_t &nnz);

/*  read coordinate data */

//only read along the major dimension

void mm_read_crd_sym_data(std::ifstream &mtx_file, 
                          int64_t M, int64_t N, int64_t &nnz, 
                          std::vector<std::vector<int>> &idx, 
                          std::vector<std::vector<float>> &val, 
                          enum MM_DTYPE_CODE mtx_data_type); 

void mm_read_crd_general_data(std::ifstream &mtx_file, 
                              int64_t M, int64_t N, int64_t &nnz, 
                              std::vector<std::vector<int>> &idx, std::vector<std::vector<float>> &val, 
                              enum MM_DTYPE_CODE mtx_data_type);  

/*  high level I/O routines */

/**
 * @property: for pattern matrices, the values are set to 1
 * @property: The input mtx files are col-major
 * @property: Banner should be read first
**/
int mm_read_mtx_crd(std::ifstream &mtx_file, 
                    int64_t &M, int64_t &N, int64_t &nnz, 
					std::vector<std::vector<int>> &idx, 
					std::vector<std::vector<float>> &val, 
					enum MM_MTYPE_CODE &mtx_mat_type, 
					enum MM_DTYPE_CODE &mtx_data_type,
					enum MM_SYM_COMPRESS_FLAG mtx_sym_compress_flag);

void mm_mtx_to_csr(std::vector<std::vector<int>> &mtx_idx,
                   std::vector<std::vector<float>> &mtx_val,
                   int* csr_offsets, int* csr_colid, 
                   float* csr_val, int64_t nnz);

#endif
