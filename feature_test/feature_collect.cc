/**
 * Collect the features among mtxs
**/

#include "../../comparison/cpu_spmm/mkl_spmm/utils/base.h"
#include "./feature_generation.h"
#include "../../comparison/cpu_spmm/mkl_spmm/utils/mtxio.h"

#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>

int feature_block_num_selector(int64_t N)
{
    //select feature num
    int feature_block_num;
    if (N / 16 < 64)
    {
        return 0;
    }
    else
    {
        if (N /16 >= 1024)
        {
            feature_block_num = 1024;
        }
        else
        {
            if (N / 16 >= 512)
            {
                feature_block_num = 512;
            }
            else
            {
                if (N / 16 >= 256)
                {
                    feature_block_num = 256;
                }
                else
                {
                    if (N / 16 >= 128)
                    {
                        feature_block_num = 128;
                    }
                    else
                    {
                        feature_block_num = 64;
                    }
                }
            }
        }
    }
    return feature_block_num;
}

void mtx_feature_collector(std::string test_file_name)
{
    enum MM_SYM_COMPRESS_FLAG mtx_sym_compress_flag = MM_SYM_UNCOMPRESS;

    //initialize the mtx variables
    enum MM_MTYPE_CODE mtx_mat_type;
    enum MM_DTYPE_CODE mtx_data_type;
    int mat_support_flag;
    int64_t M, N, nnz;
    int64_t K = MM_DEFAULT_DENSE_MAT_COL;  
    std::vector<std::vector<int>> idx;
    std::vector<std::vector<float>> val;

    int feature_block_num = 1024;

    struct timeval cpu_start, cpu_compute_time;
    double cpu_time;

    std::ifstream mtx_file(test_file_name);

    mm_read_banner(mtx_file, 
                   mtx_mat_type, mtx_data_type,
                   mat_support_flag);
    
    if (mat_support_flag == 0)
    {
        std::cout << "unsupported matrix type, exit..." << std::endl;
        exit(-1);
    }
    
    mm_read_mtx_crd(mtx_file, M, N, nnz, idx, val, 
                    mtx_mat_type, mtx_data_type, mtx_sym_compress_flag);

    feature_block_num = feature_block_num_selector(N);
    if (feature_block_num == 0)
    {
        return;
    }

    SpMMFeature::MatrixFeature *mat_feature = (SpMMFeature::MatrixFeature*)malloc(sizeof(SpMMFeature::MatrixFeature));

    SpMMFeature::matrix_feature_malloc(mat_feature, feature_block_num);

    gettimeofday(&cpu_start, NULL);
    SpMMFeature::matrix_feature_generate(mat_feature, 0, N, N, M, K, nnz, feature_block_num, idx, val);
    gettimeofday(&cpu_compute_time, NULL);

    cpu_time = ((double)(cpu_compute_time.tv_sec - cpu_start.tv_sec))*1000.0 + ((double)(cpu_compute_time.tv_usec - cpu_start.tv_usec))/1000.0;
    std::cout << "Execution: " << cpu_time << " ms" << std::endl;
      
    SpMMFeature::matrix_feature_output(mat_feature);

    mtx_file.close();
    matrix_feature_free(mat_feature);

    return;
}

int main(int argc, char* argv[])
{
      
    std::string test_file_name = argv[1];
    mtx_feature_collector(test_file_name);
    //mat_feature_collector(test_file_name, mat_type);
    
    return 0;
}