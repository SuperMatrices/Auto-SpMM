/**
 * Generate features of matrices and system for models
 * Include c++ functions
 * Pybind11 binding code is only in feature_generation.cc
**/

#ifndef SPMM_MODEL_FEATURE_GENERATION_H
#define SPMM_MODEL_FEATURE_GENERATION_H

#pragma once

#include "../../comparison/cpu_spmm/mkl_spmm/utils/base.h"
#define FEATURE_BLOCK_NUM 256

namespace SpMMFeature 
{ 
    //features affect performance in tc
    //var:deviation(sqrt(variance))
    struct TCBlockFeature
    {
        int64_t max_tc_block_num;
        int64_t min_tc_block_num;
        int64_t ave_tc_block_num;
        double var_tc_block_num;
        double gini_tc_block_num;

        double max_padding_ratio;
        double min_padding_ratio;
        double ave_padding_ratio;
        double var_padding_ratio;
        double gini_padding_ratio;             
    };

    //features affect performance in cc
    struct CCBlockFeature
    {
        int64_t max_row_nnz_num;
        int64_t min_row_nnz_num;
        int64_t ave_row_nnz_num;
        double var_row_nnz_num;
        double gini_row_nnz_num;

        double max_var_pos_row_nnz;  //the variance of colid within a row
        double min_var_pos_row_nnz;
        double ave_var_pos_row_nnz;
        double var_var_pos_row_nnz;

        //for groups 
        int64_t max_8_row_nnz_num;
        int64_t min_8_row_nnz_num;
        int64_t ave_8_row_nnz_num;
        double ave_8_var_pos_row_nnz;
        
        int64_t max_16_row_nnz_num;
        int64_t min_16_row_nnz_num;
        int64_t ave_16_row_nnz_num;
        double ave_16_var_pos_row_nnz;

        //int64_t max_32_row_nnz_num;
        //int64_t min_32_row_nnz_num;
        //int64_t ave_32_row_nnz_num;
        //double ave_32_var_pos_row_nnz;
    };

    //feature within a block
    struct BlockFeature
    {
        int64_t nnz;
        
        TCBlockFeature* tc_feature;
        CCBlockFeature* cc_feature;
    };
       
    struct MatrixFeature
    {               
        //mat dims
        int64_t M;
        int64_t N;
        int64_t K;

        int64_t nnz;                  

        int64_t block_feature_list_len;
        BlockFeature** block_features; 

        //total matrix feature
        double density;         
        int64_t max_row_nnz_num;
        int64_t min_row_nnz_num;
        int64_t ave_row_nnz_num;
       
        //summarized block feature  
        int64_t ave_nnz_per_block;
        int64_t ave_max_tc_block_num_per_block;
        int64_t ave_min_tc_block_num_per_block;
        int64_t ave_tc_block_num_per_block;
        double var_ave_tc_block_num_per_block;
        double ave_var_tc_block_num_per_block;
        double ave_gini_tc_block_num;  //the gini coefficient of gini_tc_block_num

        double ave_max_padding_ratio_per_block;
        double ave_min_padding_ratio_per_block;
        double ave_ave_padding_ratio_per_block;
        double var_ave_padding_ratio_per_block;
        double ave_var_padding_ratio_per_block;
        double ave_gini_padding_ratio_per_block;

        int64_t ave_max_row_nnz_num_per_block;
        int64_t ave_min_row_nnz_num_per_block;
        int64_t ave_ave_row_nnz_num_per_block;
        double var_ave_row_nnz_num_per_block;
        double ave_var_row_nnz_num_per_block;
        double ave_gini_row_nnz_num_per_block; 

        double ave_max_var_pos_row_nnz_per_block;
        double ave_min_var_pos_row_nnz_per_block;
        double ave_ave_var_pos_row_nnz_per_block;
        double var_ave_var_pos_row_nnz_per_block;
        double ave_var_var_pos_row_nnz_per_block;
        
        int64_t ave_max_8_row_nnz_num_per_block;
        int64_t ave_min_8_row_nnz_num_per_block;
        int64_t ave_ave_8_row_nnz_num_per_block;
        double ave_ave_8_var_pos_row_nnz_per_block;
        
        int64_t ave_max_16_row_nnz_num_per_block;
        int64_t ave_min_16_row_nnz_num_per_block;
        int64_t ave_ave_16_row_nnz_num_per_block;
        double ave_ave_16_var_pos_row_nnz_per_block;

        //int64_t ave_max_32_row_nnz_num_per_block;
        //int64_t ave_min_32_row_nnz_num_per_block;
        //int64_t ave_ave_32_row_nnz_num_per_block;
        //double ave_ave_32_var_pos_row_nnz_per_block;

    }; 

    //memory in MB
    struct SystemFeature
    {
        int64_t gpu_global_memory_size;        
    };

    //generate parallelly
    void matrix_feature_initialize(MatrixFeature *mat_feature,               
                                   int64_t start_idx, 
                                   int64_t end_idx,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t nnz,
                                   int64_t block_num,
                                   std::vector<std::vector<int>> & idx,
                                   std::vector<std::vector<float>> &val
    );

    void matrix_feature_malloc(MatrixFeature *mat_feature,
                               int64_t block_num);
    
    void matrix_feature_generate(MatrixFeature *mat_feature,               
                                int64_t start_idx, 
                                int64_t end_idx,
                                int64_t M, int64_t N, int64_t K,
                                int64_t nnz,
                                int64_t block_num,
                                std::vector<std::vector<int>> & idx,
                                std::vector<std::vector<float>> &val);
    
    void system_feature_initialize(SystemFeature *sys_feature, int64_t gpu_global_memory_size);

    //for visualization and debug
    void block_feature_display(TCBlockFeature *tc_feature,
                               CCBlockFeature *cc_feature);
    void matrix_feature_display(MatrixFeature *mat_feature);
    //only output the summed 
    void matrix_feature_output(MatrixFeature *mat_feature);
    void system_feature_display(SystemFeature *sys_feature);

    void matrix_feature_free(MatrixFeature* mat_feature);
    void system_feature_free(SystemFeature* sys_feature);

}


#endif
