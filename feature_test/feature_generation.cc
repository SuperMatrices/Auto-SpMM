
#include "./feature_generation.h"
#include <algorithm>
#include <cmath>

namespace SpMMFeature
{
    void block_feature_initialize(TCBlockFeature *tc_feature, 
                                  CCBlockFeature *cc_feature, 
                                  std::vector<std::vector<int>> &idx, 
                                  std::vector<std::vector<float>> &val)
    {
        tc_feature->max_tc_block_num = 0;
        tc_feature->min_tc_block_num = 9e8;
        tc_feature->ave_tc_block_num = 0;
        tc_feature->var_tc_block_num = 0;
        tc_feature->gini_tc_block_num = 1;

        tc_feature->max_padding_ratio = 0;
        tc_feature->min_padding_ratio = 1;
        tc_feature->ave_padding_ratio = 0;
        tc_feature->var_padding_ratio = 0;
        tc_feature->gini_padding_ratio = 1;        

        cc_feature->max_row_nnz_num = 0;
        cc_feature->min_row_nnz_num = 9e8;
        cc_feature->ave_row_nnz_num = 0;
        cc_feature->var_row_nnz_num = 0;
        cc_feature->gini_row_nnz_num = 1;

        cc_feature->max_var_pos_row_nnz = 0;
        cc_feature->min_var_pos_row_nnz = 9e8;
        cc_feature->ave_var_pos_row_nnz = 0;
        cc_feature->var_var_pos_row_nnz = 0;

        cc_feature->max_8_row_nnz_num = 0;
        cc_feature->min_8_row_nnz_num = 9e8;
        cc_feature->ave_8_row_nnz_num = 0;
        cc_feature->ave_8_var_pos_row_nnz = 0;

        cc_feature->max_16_row_nnz_num = 0;
        cc_feature->min_16_row_nnz_num = 9e8;
        cc_feature->ave_16_row_nnz_num = 0;
        cc_feature->ave_16_var_pos_row_nnz = 0;

        return;
    }

    void summed_feature_initialize(MatrixFeature *mat_feature)
    {
        mat_feature->max_row_nnz_num = 0;
        mat_feature->min_row_nnz_num = 9e8;
        //mat_feature->ave_row_nnz_num = 0;
               
        //mat_feature->ave_nnz_per_block = 0;
        mat_feature->ave_max_tc_block_num_per_block = 0;
        mat_feature->ave_min_tc_block_num_per_block = 0;
        mat_feature->ave_tc_block_num_per_block = 0;
        mat_feature->var_ave_tc_block_num_per_block = 0;
        mat_feature->ave_var_tc_block_num_per_block = 0;
        mat_feature->ave_gini_tc_block_num = 0;  //the gini coefficient of gini_tc_block_num

        mat_feature->ave_max_padding_ratio_per_block = 0;
        mat_feature->ave_min_padding_ratio_per_block = 0;
        mat_feature->ave_ave_padding_ratio_per_block = 0;
        mat_feature->var_ave_padding_ratio_per_block = 0;
        mat_feature->ave_var_padding_ratio_per_block = 0;
        mat_feature->ave_gini_padding_ratio_per_block = 0;

        mat_feature->ave_max_row_nnz_num_per_block = 0;
        mat_feature->ave_min_row_nnz_num_per_block = 0;
        mat_feature->ave_ave_row_nnz_num_per_block = 0;
        mat_feature->var_ave_row_nnz_num_per_block = 0;
        mat_feature->ave_var_row_nnz_num_per_block = 0;
        mat_feature->ave_gini_row_nnz_num_per_block = 0; 

        mat_feature->ave_max_var_pos_row_nnz_per_block = 0;
        mat_feature->ave_min_var_pos_row_nnz_per_block = 0;
        mat_feature->ave_ave_var_pos_row_nnz_per_block = 0;
        mat_feature->var_ave_var_pos_row_nnz_per_block = 0;
        mat_feature->ave_var_var_pos_row_nnz_per_block = 0;

        mat_feature->ave_max_8_row_nnz_num_per_block = 0;
        mat_feature->ave_min_8_row_nnz_num_per_block = 0;
        mat_feature->ave_ave_8_row_nnz_num_per_block = 0;
        mat_feature->ave_ave_8_var_pos_row_nnz_per_block = 0;
        
        mat_feature->ave_max_16_row_nnz_num_per_block = 0;
        mat_feature->ave_min_16_row_nnz_num_per_block = 0;
        mat_feature->ave_ave_16_row_nnz_num_per_block = 0;
        mat_feature->ave_ave_16_var_pos_row_nnz_per_block = 0;
    }

    void block_feature_display(TCBlockFeature *tc_feature,
                               CCBlockFeature *cc_feature)
    {
        std::cout << "TC Block Feature: " << std::endl;
        std::cout << tc_feature->max_tc_block_num << " "
                  << tc_feature->min_tc_block_num << " "
                  << tc_feature->ave_tc_block_num << " "
                  << tc_feature->var_tc_block_num << " "
                  << tc_feature->gini_tc_block_num << " "

                  << tc_feature->max_padding_ratio << " "
                  << tc_feature->min_padding_ratio << " "
                  << tc_feature->ave_padding_ratio << " "
                  << tc_feature->var_padding_ratio << " "
                  << tc_feature->gini_padding_ratio << " " 
                  << std::endl;

        std::cout << "CC Block Feature: " << std::endl;
        std::cout << cc_feature->max_row_nnz_num << " "
                  << cc_feature->min_row_nnz_num << " "
                  << cc_feature->ave_row_nnz_num << " "
                  << cc_feature->var_row_nnz_num << " "
                  << cc_feature->gini_row_nnz_num << " "

                  << cc_feature->max_var_pos_row_nnz << " "  
                  << cc_feature->min_var_pos_row_nnz << " "
                  << cc_feature->ave_var_pos_row_nnz << " "
                  << cc_feature->var_var_pos_row_nnz << " "

                  << cc_feature->max_8_row_nnz_num << " "
                  << cc_feature->min_8_row_nnz_num << " "
                  << cc_feature->ave_8_row_nnz_num << " "
                  << cc_feature->ave_8_var_pos_row_nnz << " "

                  << cc_feature->max_16_row_nnz_num << " "
                  << cc_feature->min_16_row_nnz_num << " "
                  << cc_feature->ave_16_row_nnz_num << " "
                  << cc_feature->ave_16_var_pos_row_nnz << " "
                  << std::endl;
        return;
    }

    //the gini section: (<0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1), (1,2), (2, 5), (5, 10), (> 10)
    void gini_counter_append(std::vector<double> &gini_counter, 
                             double data, 
                             double standard)
    {
        double ratio = data / standard;
        if (ratio < 0.1)
        {
            gini_counter[0]++;
            return;
        }

        if (ratio >= 0.1 && ratio < 0.2)
        {
            gini_counter[1]++;
            return;
        }

        if (ratio >= 0.2 && ratio < 0.5)
        {
            gini_counter[2]++;
            return;
        }

        if (ratio >= 0.5 && ratio < 1)
        {
            gini_counter[3]++;
            return;
        }

        if (ratio >=1 && ratio < 2)
        {
            gini_counter[4]++;
            return;
        }

        if (ratio >=2 && ratio < 5)
        {
            gini_counter[5]++;
            return;
        }

        if (ratio >= 5 && ratio < 10)
        {
            gini_counter[6]++;
            return;
        }

        if (ratio >= 10)
        {
            gini_counter[7]++;
        }

        return;
    }
   
    //feature generator
    void mat_block_feature_generate(TCBlockFeature* tc_feature,
                                    CCBlockFeature* cc_feature,
                                    int64_t start_row_id, 
                                    int64_t end_row_id,
                                    std::vector<std::vector<int>> &idx, 
                                    std::vector<std::vector<float>> &val
                                    )
    {
        int64_t row_len = end_row_id - start_row_id + 1;
                
        std::vector<int64_t> tc_block_num;
        tc_block_num.resize((row_len + 15) / 16);

        std::vector<double> tc_padding;
        tc_padding.resize((row_len + 15) / 16);

        std::vector<int64_t> cc_8_row_nnz;
        cc_8_row_nnz.resize((row_len + 7) / 8);
        std::fill(cc_8_row_nnz.begin(), cc_8_row_nnz.end(), 0);

        std::vector<int64_t> cc_16_row_nnz;
        cc_16_row_nnz.resize((row_len + 15) / 16);
        std::fill(cc_16_row_nnz.begin(), cc_16_row_nnz.end(), 0);

        std::vector<double> cc_var_pos_row_nnz;
        cc_var_pos_row_nnz.resize(row_len);
        std::fill(cc_var_pos_row_nnz.begin(), cc_var_pos_row_nnz.end(), 0);

        std::vector<double> gini_counter(8, 0);
        std::vector<double> gini_counter2(8, 0);
        
        int64_t min_col_id = 9e8, max_col_id = 0, ave_col_id = 0;
        int64_t total_nnz = 0;        
        int64_t sub_16_block_counter = 0, sub_8_block_counter = 0;

        block_feature_initialize(tc_feature, cc_feature, idx, val);

        for (int64_t i = start_row_id; i <= end_row_id; i+=16)
        {
            min_col_id = 9e8;
            max_col_id = 0;
            total_nnz = 0;

            #pragma unroll
            for (int64_t j = 0; j < 16 && i + j <= end_row_id; j++)
            {
                //for tc
                if (idx[i + j].size() > 0)
                {
                    if (idx[i + j][0] < min_col_id)
                    {
                        min_col_id = idx[i + j][0];
                    }
                
                    if (idx[i + j][idx[i + j].size()-1] > max_col_id)
                    {
                        max_col_id = idx[i + j][idx[i + j].size()-1];
                    }
                }
                else
                {
                    min_col_id = 0;                    
                }

                //for cc: single row
                if (idx[i + j].size() > cc_feature->max_row_nnz_num)
                {
                    cc_feature->max_row_nnz_num = idx[i + j].size();                    
                }
                
                if (idx[i + j].size() < cc_feature->min_row_nnz_num)
                {
                    cc_feature->min_row_nnz_num = idx[i + j].size();
                }

                cc_feature->ave_row_nnz_num += idx[i + j].size();

                cc_var_pos_row_nnz[i + j - start_row_id] = 0;
                ave_col_id = 0;
                for (int64_t k = 0; k < idx[i + j].size(); k++)
                {
                    ave_col_id += idx[i + j][k];                    
                }

                if (idx[i + j].size() > 0)
                {
                    ave_col_id = ave_col_id / idx[i + j].size();
                }
                else
                {
                    ave_col_id = 0;
                }

                for (int64_t k = 0; k < idx[i + j].size(); k++)
                {
                    cc_var_pos_row_nnz[i + j-start_row_id] += (idx[i + j][k] - ave_col_id) * (idx[i + j][k] - ave_col_id);
                }

                if (idx[i + j].size() > 0)
                {
                    cc_var_pos_row_nnz[i + j - start_row_id] /= idx[i + j].size();
                    cc_var_pos_row_nnz[i + j - start_row_id] = sqrt(cc_var_pos_row_nnz[i + j - start_row_id]);
                }
                else
                {
                    cc_var_pos_row_nnz[i + j - start_row_id] = 0;
                }

                //for cc : 8 row
                if (j < 8)
                {
                    cc_8_row_nnz[sub_8_block_counter] += idx[i + j].size();                    
                }
                else
                {
                    cc_8_row_nnz[sub_8_block_counter+1] += idx[i + j].size();
                }

                //for cc : 16 row
                cc_16_row_nnz[sub_16_block_counter] += idx[i + j].size();

                total_nnz += idx[i + j].size();
            }            

            //tc_block_num
            tc_block_num[sub_16_block_counter] = (max_col_id - min_col_id + 1 + 15) / 16;

            //padding_ratio
            tc_padding[sub_16_block_counter] = 1 - ((double)total_nnz)/(tc_block_num[sub_16_block_counter] * 16 * 16);
            
            sub_16_block_counter++;
            sub_8_block_counter += 2;
        }       

        sub_16_block_counter = (row_len + 15) / 16;
        sub_8_block_counter = (row_len + 7) / 8;   

        cc_feature->ave_row_nnz_num = cc_feature->ave_row_nnz_num / row_len;  

        //features in single row
        double tmp_8_var_pos_row_nnz = 0, tmp_16_var_pos_row_nnz = 0;
        for (int64_t i = 0; i < row_len; i++)
        {
            //var_pos
            if (cc_var_pos_row_nnz[i] < cc_feature->min_var_pos_row_nnz)
            {
                cc_feature->min_var_pos_row_nnz = cc_var_pos_row_nnz[i];
            }

            if (cc_var_pos_row_nnz[i] > cc_feature->max_var_pos_row_nnz)
            {
                cc_feature->max_var_pos_row_nnz = cc_var_pos_row_nnz[i];
            }

            cc_feature->ave_var_pos_row_nnz += cc_var_pos_row_nnz[i];

            if (i % 8 == 0)
            {
                cc_feature->ave_8_var_pos_row_nnz += tmp_8_var_pos_row_nnz / 8;
                tmp_8_var_pos_row_nnz = 0;
            }

            if (i % 16 == 0)
            {
                cc_feature->ave_16_var_pos_row_nnz += tmp_16_var_pos_row_nnz / 16;
                tmp_16_var_pos_row_nnz = 0;                
            }
            
            tmp_8_var_pos_row_nnz += cc_var_pos_row_nnz[i]; 
            tmp_16_var_pos_row_nnz += cc_var_pos_row_nnz[i];

            //row num
            cc_feature->var_row_nnz_num += (idx[start_row_id + i].size() - cc_feature->ave_row_nnz_num) * (idx[start_row_id + i].size() - cc_feature->ave_row_nnz_num);
            gini_counter_append(gini_counter, idx[start_row_id + i].size(), cc_feature->ave_row_nnz_num);
        }        
 
        cc_feature->ave_var_pos_row_nnz = cc_feature->ave_var_pos_row_nnz / row_len;
        cc_feature->ave_8_var_pos_row_nnz = cc_feature->ave_8_var_pos_row_nnz / sub_8_block_counter;
        cc_feature->ave_16_var_pos_row_nnz = cc_feature->ave_16_var_pos_row_nnz / sub_16_block_counter;
        cc_feature->var_row_nnz_num /= row_len;
        cc_feature->var_row_nnz_num = sqrt(cc_feature->var_row_nnz_num);

        for (int64_t i = 0; i < row_len; i++)
        {
            cc_feature->var_var_pos_row_nnz = (cc_var_pos_row_nnz[i] - cc_feature->ave_var_pos_row_nnz) * (cc_var_pos_row_nnz[i] - cc_feature->ave_var_pos_row_nnz);
        }

        cc_feature->var_var_pos_row_nnz /= row_len;
        cc_feature->var_var_pos_row_nnz = sqrt(cc_feature->var_var_pos_row_nnz);

        for (int64_t i = 0; i < 8; i++)
        {
            cc_feature->gini_row_nnz_num -= (gini_counter[i]/row_len) * (gini_counter[i]/row_len);
        }

        //feature in 8 row     
        for (int64_t i = 0; i < sub_8_block_counter; i++)
        {
            if (cc_8_row_nnz[i] > cc_feature->max_8_row_nnz_num)
            {
                cc_feature->max_8_row_nnz_num = cc_8_row_nnz[i];
            } 

            if (cc_8_row_nnz[i] < cc_feature->min_8_row_nnz_num)
            {
                cc_feature->min_8_row_nnz_num = cc_8_row_nnz[i];
            }  

            cc_feature->ave_8_row_nnz_num += cc_8_row_nnz[i];
        }        

        cc_feature->ave_8_row_nnz_num = cc_feature->ave_8_row_nnz_num / sub_8_block_counter;
        
        //feature in 16 row
        for (int64_t i = 0; i < sub_16_block_counter; i++)
        {
            if (tc_block_num[i] > tc_feature->max_tc_block_num)
            {
                tc_feature->max_tc_block_num = tc_block_num[i];
            }

            if (tc_block_num[i] < tc_feature->min_tc_block_num)
            {
                tc_feature->min_tc_block_num = tc_block_num[i];
            }

            tc_feature->ave_tc_block_num += tc_block_num[i];

            if (tc_padding[i] > tc_feature->max_padding_ratio)
            {
                tc_feature->max_padding_ratio = tc_padding[i];
            }

            if (tc_padding[i] < tc_feature->min_padding_ratio)
            {
                tc_feature->min_padding_ratio = tc_padding[i];                
            }

            tc_feature->ave_padding_ratio += tc_padding[i];

            if (cc_16_row_nnz[i] > cc_feature->max_16_row_nnz_num)
            {
                cc_feature->max_16_row_nnz_num = cc_16_row_nnz[i];
            }

            if (cc_16_row_nnz[i] < cc_feature->min_16_row_nnz_num)
            {
                cc_feature->min_16_row_nnz_num = cc_16_row_nnz[i];
            }

            cc_feature->ave_16_row_nnz_num += cc_16_row_nnz[i];
        }
 
        tc_feature->ave_tc_block_num = tc_feature->ave_tc_block_num / sub_16_block_counter;
        tc_feature->ave_padding_ratio = tc_feature->ave_padding_ratio / sub_16_block_counter;
        
        cc_feature->ave_16_row_nnz_num = cc_feature->ave_16_row_nnz_num / sub_16_block_counter;

        //the var and gini in 16 block row
        std::fill(gini_counter.begin(), gini_counter.end(), 0);
        for (int64_t i = 0; i < sub_16_block_counter; i++)
        {
            tc_feature->var_tc_block_num += (tc_block_num[i] - tc_feature->ave_tc_block_num) * (tc_block_num[i] - tc_feature->ave_tc_block_num);

            gini_counter_append(gini_counter, tc_block_num[i], tc_feature->ave_tc_block_num);

            tc_feature->var_padding_ratio += (tc_padding[i] - tc_feature->ave_padding_ratio) * (tc_padding[i] - tc_feature->ave_padding_ratio);

            gini_counter_append(gini_counter2, tc_padding[i], tc_feature->ave_padding_ratio);
        }

        tc_feature->var_tc_block_num /= sub_16_block_counter;
        tc_feature->var_tc_block_num = sqrt(tc_feature->var_tc_block_num);
        tc_feature->var_padding_ratio /= sub_16_block_counter;
        tc_feature->var_padding_ratio = sqrt(tc_feature->var_padding_ratio);

        for (int64_t i = 0; i < 8; i++)
        {
            tc_feature->gini_tc_block_num -= (gini_counter[i]/sub_16_block_counter) * (gini_counter[i]/sub_16_block_counter);
            tc_feature->gini_padding_ratio -= (gini_counter2[i]/sub_16_block_counter) * (gini_counter2[i]/sub_16_block_counter);
        }
        
        return;
    }

    void mat_summed_feature_generate(MatrixFeature *mat_feature)
    {
        //total feature
        mat_feature->density = (double)mat_feature->nnz / (double)mat_feature->M / (double)mat_feature->N;
        mat_feature->ave_row_nnz_num = mat_feature->nnz / mat_feature->M;
                
        int64_t block_len = mat_feature->block_feature_list_len;
        mat_feature->ave_nnz_per_block = mat_feature->nnz / block_len;

        BlockFeature** block_features = mat_feature->block_features;
        CCBlockFeature* cc_feature;
        TCBlockFeature* tc_feature;
        
        summed_feature_initialize(mat_feature);

        for (int64_t i = 0; i < block_len; i++)
        {
            cc_feature = (block_features[i])->cc_feature;
            tc_feature = (block_features[i])->tc_feature;

            if (cc_feature->max_row_nnz_num > mat_feature->max_row_nnz_num)
            {
                mat_feature->max_row_nnz_num = cc_feature->max_row_nnz_num;
            }            

            if (cc_feature->min_row_nnz_num < mat_feature->min_row_nnz_num)
            {
                mat_feature->min_row_nnz_num = cc_feature->min_row_nnz_num;
            }

            mat_feature->ave_max_tc_block_num_per_block += tc_feature->max_tc_block_num;
            mat_feature->ave_min_tc_block_num_per_block += tc_feature->min_tc_block_num;
            mat_feature->ave_tc_block_num_per_block += tc_feature->ave_tc_block_num;
            mat_feature->ave_var_tc_block_num_per_block += tc_feature->var_tc_block_num;
            mat_feature->ave_gini_tc_block_num += tc_feature->gini_tc_block_num;

            mat_feature->ave_max_padding_ratio_per_block += tc_feature->max_padding_ratio;
            mat_feature->ave_min_padding_ratio_per_block += tc_feature->min_padding_ratio;
            mat_feature->ave_ave_padding_ratio_per_block += tc_feature->ave_padding_ratio;
            mat_feature->ave_var_padding_ratio_per_block += tc_feature->var_padding_ratio;
            mat_feature->ave_gini_padding_ratio_per_block += tc_feature->gini_padding_ratio;

            mat_feature->ave_max_row_nnz_num_per_block += cc_feature->max_row_nnz_num;
            mat_feature->ave_min_row_nnz_num_per_block += cc_feature->min_row_nnz_num;
            mat_feature->ave_ave_row_nnz_num_per_block += cc_feature->ave_row_nnz_num;
            mat_feature->ave_var_row_nnz_num_per_block += cc_feature->var_row_nnz_num;
            mat_feature->ave_gini_row_nnz_num_per_block += cc_feature->gini_row_nnz_num;

            mat_feature->ave_max_var_pos_row_nnz_per_block += cc_feature->max_var_pos_row_nnz;
            mat_feature->ave_min_var_pos_row_nnz_per_block += cc_feature->min_var_pos_row_nnz;
            mat_feature->ave_ave_var_pos_row_nnz_per_block += cc_feature->ave_var_pos_row_nnz;
            mat_feature->ave_var_var_pos_row_nnz_per_block += cc_feature->var_var_pos_row_nnz;

            mat_feature->ave_max_8_row_nnz_num_per_block += cc_feature->max_8_row_nnz_num;
            mat_feature->ave_min_8_row_nnz_num_per_block += cc_feature->min_8_row_nnz_num;
            mat_feature->ave_ave_8_row_nnz_num_per_block += cc_feature->ave_8_row_nnz_num;
            mat_feature->ave_ave_8_var_pos_row_nnz_per_block += cc_feature->ave_8_var_pos_row_nnz;

            mat_feature->ave_max_16_row_nnz_num_per_block += cc_feature->max_16_row_nnz_num;
            mat_feature->ave_min_16_row_nnz_num_per_block += cc_feature->min_16_row_nnz_num;
            mat_feature->ave_ave_16_row_nnz_num_per_block += cc_feature->ave_16_row_nnz_num;
            mat_feature->ave_ave_16_var_pos_row_nnz_per_block += cc_feature->ave_16_var_pos_row_nnz;
        }

        mat_feature->ave_max_tc_block_num_per_block /= block_len;
        mat_feature->ave_min_tc_block_num_per_block /= block_len;
        mat_feature->ave_tc_block_num_per_block /= block_len;
        mat_feature->ave_var_tc_block_num_per_block /= block_len;
        mat_feature->ave_gini_tc_block_num /= block_len; 

        mat_feature->ave_max_padding_ratio_per_block /= block_len;
        mat_feature->ave_min_padding_ratio_per_block /= block_len;
        mat_feature->ave_ave_padding_ratio_per_block /= block_len;
        mat_feature->ave_var_padding_ratio_per_block /= block_len;
        mat_feature->ave_gini_padding_ratio_per_block /= block_len;

        mat_feature->ave_max_row_nnz_num_per_block /= block_len;
        mat_feature->ave_min_row_nnz_num_per_block /= block_len;
        mat_feature->ave_ave_row_nnz_num_per_block /= block_len;
        mat_feature->ave_var_row_nnz_num_per_block /= block_len;
        mat_feature->ave_gini_row_nnz_num_per_block /= block_len; 

        mat_feature->ave_max_var_pos_row_nnz_per_block /= block_len;
        mat_feature->ave_min_var_pos_row_nnz_per_block /= block_len;
        mat_feature->ave_ave_var_pos_row_nnz_per_block /= block_len;
        mat_feature->ave_var_var_pos_row_nnz_per_block /= block_len;

        mat_feature->ave_max_8_row_nnz_num_per_block /= block_len;
        mat_feature->ave_min_8_row_nnz_num_per_block /= block_len;
        mat_feature->ave_ave_8_row_nnz_num_per_block /= block_len;
        mat_feature->ave_ave_8_var_pos_row_nnz_per_block /= block_len;
        
        mat_feature->ave_max_16_row_nnz_num_per_block /= block_len;
        mat_feature->ave_min_16_row_nnz_num_per_block /= block_len;
        mat_feature->ave_ave_16_row_nnz_num_per_block /= block_len;
        mat_feature->ave_ave_16_var_pos_row_nnz_per_block /= block_len;

        //the vars
        for (int64_t i = 0; i < block_len; i++)
        {
            cc_feature = (block_features[i])->cc_feature;
            tc_feature = (block_features[i])->tc_feature;

            mat_feature->var_ave_tc_block_num_per_block += (tc_feature->ave_tc_block_num - mat_feature->ave_tc_block_num_per_block) * (tc_feature->ave_tc_block_num - mat_feature->ave_tc_block_num_per_block);

            mat_feature->var_ave_padding_ratio_per_block += (tc_feature->ave_padding_ratio - mat_feature->ave_ave_padding_ratio_per_block) * (tc_feature->ave_padding_ratio - mat_feature->ave_ave_padding_ratio_per_block);

            mat_feature->var_ave_row_nnz_num_per_block += (cc_feature->ave_row_nnz_num - mat_feature->ave_ave_row_nnz_num_per_block) * (cc_feature->ave_row_nnz_num - mat_feature->ave_ave_row_nnz_num_per_block);

            mat_feature->var_ave_var_pos_row_nnz_per_block +=
            (cc_feature->ave_var_pos_row_nnz - mat_feature->ave_ave_var_pos_row_nnz_per_block) * (cc_feature->ave_var_pos_row_nnz - mat_feature->ave_ave_var_pos_row_nnz_per_block);       
        }

        mat_feature->var_ave_tc_block_num_per_block /= block_len;
        mat_feature->var_ave_tc_block_num_per_block= sqrt(mat_feature->var_ave_tc_block_num_per_block);
        mat_feature->var_ave_padding_ratio_per_block /= block_len;
        mat_feature->var_ave_padding_ratio_per_block = sqrt(mat_feature->var_ave_padding_ratio_per_block);
        mat_feature->var_ave_row_nnz_num_per_block /= block_len;
        mat_feature->var_ave_row_nnz_num_per_block = sqrt(mat_feature->var_ave_row_nnz_num_per_block);
        mat_feature->var_ave_var_pos_row_nnz_per_block /= block_len;
        mat_feature->var_ave_var_pos_row_nnz_per_block = sqrt(mat_feature->var_ave_var_pos_row_nnz_per_block);

        return;
    }
     

    void matrix_feature_initialize(MatrixFeature *mat_feature,
                                   int64_t start_idx, int64_t end_idx,
                                   int64_t M, int64_t N, int64_t K,int64_t nnz,
                                   int64_t block_num,
                                   std::vector<std::vector<int>> &idx,
                                   std::vector<std::vector<float>> &val)
    {
        mat_feature->M = M;
        mat_feature->N = N;
        mat_feature->K = K;
        mat_feature->nnz = nnz;

        mat_feature->block_feature_list_len = block_num;

        mat_feature->block_features = (BlockFeature**)malloc(mat_feature->block_feature_list_len * sizeof(BlockFeature*));

        int64_t row_per_block = M / block_num;
        int64_t res_block_num = M - block_num * row_per_block;

        for (int64_t i = 0; i < block_num; i++)
        {
            (mat_feature->block_features)[i] = (BlockFeature*)malloc(sizeof(BlockFeature));
            BlockFeature* cur_block_feature = (mat_feature->block_features)[i];            
            cur_block_feature->tc_feature = (TCBlockFeature*)malloc(sizeof(TCBlockFeature));
            cur_block_feature->cc_feature = (CCBlockFeature*)malloc(sizeof(CCBlockFeature));            
        }

        //compute the block features
        #pragma omp parallel for num_threads(32)
        for (int64_t i = 0; i < block_num; i++)
        {
            BlockFeature* cur_block_feature = mat_feature->block_features[i];

            //int my_id = omp_get_thread_num();
            //std::cout << "my id " << my_id << std::endl;            
            int64_t start_row_id, end_row_id;

            if (i < res_block_num)
            {
                start_row_id = (row_per_block + 1) * i + start_idx;            
                end_row_id = std::min((row_per_block + 1) * (i + 1) - 1, (int64_t)idx.size() - 1) + start_idx;
            }
            else
            {
                start_row_id = (row_per_block + 1) * res_block_num + row_per_block * (i - res_block_num) + start_idx;
                end_row_id = std::min(start_row_id + row_per_block - 1, (int64_t)idx.size() - 1) + start_idx;
            }

            cur_block_feature->nnz = 0;            

            for (int64_t i = start_row_id; i <= end_row_id; i++)
            {
                cur_block_feature->nnz += idx[i].size();
            }

            mat_block_feature_generate(cur_block_feature->tc_feature,
                                       cur_block_feature->cc_feature,
                                       start_row_id, end_row_id,
                                       idx, val
                                      );    
            //block_feature_display(cur_block_feature->tc_feature,
            //                      cur_block_feature->cc_feature);         
        }

        //compute the average features
        mat_summed_feature_generate(mat_feature);       

        return;
    }

    void matrix_feature_malloc(MatrixFeature *mat_feature,
                               int64_t block_num)
    {
        mat_feature->block_feature_list_len = block_num;

        mat_feature->block_features = (BlockFeature**)malloc(mat_feature->block_feature_list_len * sizeof(BlockFeature*));

        for (int64_t i = 0; i < block_num; i++)
        {
            (mat_feature->block_features)[i] = (BlockFeature*)malloc(sizeof(BlockFeature));
            BlockFeature* cur_block_feature = (mat_feature->block_features)[i];            
            cur_block_feature->tc_feature = (TCBlockFeature*)malloc(sizeof(TCBlockFeature));
            cur_block_feature->cc_feature = (CCBlockFeature*)malloc(sizeof(CCBlockFeature));            
        }

        return;
    }
    
    void matrix_feature_generate(MatrixFeature *mat_feature,               
                                int64_t start_idx, 
                                int64_t end_idx,
                                int64_t M, int64_t N, int64_t K,
                                int64_t nnz,
                                int64_t block_num,
                                std::vector<std::vector<int>> & idx,
                                std::vector<std::vector<float>> &val)
    {
        mat_feature->M = M;
        mat_feature->N = N;
        mat_feature->K = K;
        mat_feature->nnz = nnz;

        mat_feature->block_feature_list_len = block_num;

        //std::cout << end_idx << std::endl;

        int64_t row_per_block = M / block_num;
        int64_t res_block_num = M - block_num * row_per_block;

        //compute the block features
        #pragma omp parallel for num_threads(64)
        for (int64_t i = 0; i < block_num; i++)
        {
            BlockFeature* cur_block_feature = mat_feature->block_features[i];
            
            int64_t start_row_id, end_row_id;

            if (i < res_block_num)
            {
                start_row_id = (row_per_block + 1) * i + start_idx;            
                end_row_id = std::min((row_per_block + 1) * (i + 1) - 1 + start_idx, end_idx);
            }
            else
            {
                start_row_id = (row_per_block + 1) * res_block_num + row_per_block * (i - res_block_num) + start_idx;
                end_row_id = std::min(start_row_id + row_per_block - 1, end_idx);
            }

            cur_block_feature->nnz = 0;            

            for (int64_t i = start_row_id; i <= end_row_id; i++)
            {
                cur_block_feature->nnz += idx[i].size();
            }

            mat_block_feature_generate(cur_block_feature->tc_feature,
                                       cur_block_feature->cc_feature,
                                       start_row_id, end_row_id,
                                       idx, val
                                      );    
            //block_feature_display(cur_block_feature->tc_feature,
            //                      cur_block_feature->cc_feature);         
        }

        //compute the average features
        mat_summed_feature_generate(mat_feature);
        
        return;
    }

    void system_feature_initialize(SystemFeature *sys_feature, int64_t gpu_global_memory_size)
    {
        sys_feature->gpu_global_memory_size = gpu_global_memory_size;        
        return;        
    }

    void matrix_feature_output(MatrixFeature *mat_feature)
    {
        std::cout << mat_feature->M << " "
                  << mat_feature->N << " "
                  << mat_feature->K << " "
                  << mat_feature->nnz << " "       
                  << mat_feature->density << " "         
                  << mat_feature->max_row_nnz_num << " "
                  << mat_feature->min_row_nnz_num << " "
                  << mat_feature->ave_row_nnz_num << " "
                  << mat_feature->ave_nnz_per_block << " "
                  << mat_feature->ave_max_tc_block_num_per_block << " "
                  << mat_feature->ave_min_tc_block_num_per_block << " "
                  << mat_feature->ave_tc_block_num_per_block << " "
                  << mat_feature->var_ave_tc_block_num_per_block << " "
                  << mat_feature->ave_var_tc_block_num_per_block << " "
                  << mat_feature->ave_gini_tc_block_num << " "
                  << mat_feature->ave_max_padding_ratio_per_block << " "
                  << mat_feature->ave_min_padding_ratio_per_block << " "
                  << mat_feature->ave_ave_padding_ratio_per_block << " "
                  << mat_feature->var_ave_padding_ratio_per_block << " "
                  << mat_feature->ave_var_padding_ratio_per_block << " "
                  << mat_feature->ave_gini_padding_ratio_per_block << " "
                  << mat_feature->ave_max_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_row_nnz_num_per_block << " "
                  << mat_feature->var_ave_row_nnz_num_per_block << " "
                  << mat_feature->ave_var_row_nnz_num_per_block << " "
                  << mat_feature->ave_gini_row_nnz_num_per_block << " " 
                  << mat_feature->ave_max_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_min_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_ave_var_pos_row_nnz_per_block << " "
                  << mat_feature->var_ave_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_var_var_pos_row_nnz_per_block << " "        
                  << mat_feature->ave_max_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_8_var_pos_row_nnz_per_block << " "       
                  << mat_feature->ave_max_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_16_var_pos_row_nnz_per_block << " "
                  << std::endl;
        return;
    }

    void matrix_feature_display(MatrixFeature *mat_feature)
    {
        int64_t block_len = mat_feature->block_feature_list_len;
        BlockFeature** block_features = mat_feature->block_features;
       
        std::cout << "start display" << std::endl;

        for (int64_t i = 0; i < block_len; i++)
        {
            block_feature_display((block_features[i])->tc_feature, (block_features[i])->cc_feature);           
        }

        std::cout << "The Summed Features: " << std::endl;
        std::cout << mat_feature->M << " "
                  << mat_feature->N << " "
                  << mat_feature->K << " "
                  << mat_feature->nnz << " "       
                  << mat_feature->density << " "         
                  << mat_feature->max_row_nnz_num << " "
                  << mat_feature->min_row_nnz_num << " "
                  << mat_feature->ave_row_nnz_num << " "
                  << mat_feature->ave_nnz_per_block << " "
                  << mat_feature->ave_max_tc_block_num_per_block << " "
                  << mat_feature->ave_min_tc_block_num_per_block << " "
                  << mat_feature->ave_tc_block_num_per_block << " "
                  << mat_feature->var_ave_tc_block_num_per_block << " "
                  << mat_feature->ave_var_tc_block_num_per_block << " "
                  << mat_feature->ave_gini_tc_block_num << " "
                  << mat_feature->ave_max_padding_ratio_per_block << " "
                  << mat_feature->ave_min_padding_ratio_per_block << " "
                  << mat_feature->ave_ave_padding_ratio_per_block << " "
                  << mat_feature->var_ave_padding_ratio_per_block << " "
                  << mat_feature->ave_var_padding_ratio_per_block << " "
                  << mat_feature->ave_gini_padding_ratio_per_block << " "
                  << mat_feature->ave_max_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_row_nnz_num_per_block << " "
                  << mat_feature->var_ave_row_nnz_num_per_block << " "
                  << mat_feature->ave_var_row_nnz_num_per_block << " "
                  << mat_feature->ave_gini_row_nnz_num_per_block << " " 
                  << mat_feature->ave_max_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_min_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_ave_var_pos_row_nnz_per_block << " "
                  << mat_feature->var_ave_var_pos_row_nnz_per_block << " "
                  << mat_feature->ave_var_var_pos_row_nnz_per_block << " "        
                  << mat_feature->ave_max_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_8_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_8_var_pos_row_nnz_per_block << " "       
                  << mat_feature->ave_max_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_min_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_16_row_nnz_num_per_block << " "
                  << mat_feature->ave_ave_16_var_pos_row_nnz_per_block << " "
                  << std::endl;
        return;
    }
    
    void system_feature_display(SystemFeature *sys_feature)
    {
        std::cout << "System Feature:" << std::endl;
        std::cout << sys_feature->gpu_global_memory_size 
                  << std::endl;
        return;
    }

    void matrix_feature_free(MatrixFeature* mat_feature)
    {
        for (int64_t i = 0; i < mat_feature->block_feature_list_len; i++)
        {
            free((mat_feature->block_features)[i]->tc_feature);
            free((mat_feature->block_features)[i]->cc_feature);
            free((mat_feature->block_features)[i]);
        }
        free(mat_feature->block_features);
        free(mat_feature);       
        return;
    }

    void system_feature_free(SystemFeature* sys_feature)
    {
        free(sys_feature);
    }
}