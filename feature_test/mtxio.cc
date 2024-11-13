#include "./mtxio.h"

void mm_read_banner(std::ifstream &mtx_file, 
                    enum MM_MTYPE_CODE &mtx_mat_type, 
					enum MM_DTYPE_CODE &mtx_data_type,
                    int &mat_support_flag)
{
    std::string mtx_banner;
    mat_support_flag = 1;

    //get the banner string
    getline(mtx_file, mtx_banner);

    if (mtx_banner.empty())
    {
        std::cout << "Cannot read matrix banner" << std::endl;
        return;
    }

    //check whether the mat type is supported
    const char *mtx_banner_c_str = mtx_banner.c_str();
    if (strstr(mtx_banner_c_str, "complex") != NULL || strstr(mtx_banner_c_str, "Hermitian") != NULL || strstr(mtx_banner_c_str, "array") != NULL || strstr(mtx_banner_c_str, "skew-symmetric") != NULL)
    {
        mat_support_flag = 0;
        return;
    }

    //generate the mat type code
    if (strstr(mtx_banner_c_str, "symmetric") != NULL)
    {
        mtx_mat_type = MM_SYMMETRIC;
    }

    if (strstr(mtx_banner_c_str, "general") != NULL)
    {
        mtx_mat_type = MM_GENERAL;
    }

    if (strstr(mtx_banner_c_str, "pattern") != NULL)
    {
        mtx_data_type = MM_PATTERN;
    }

    if (strstr(mtx_banner_c_str, "real") != NULL)
    {
        mtx_data_type = MM_FLOAT;
    }

    if (strstr(mtx_banner_c_str, "integer") != NULL)
    {
        mtx_data_type = MM_INTEGER;
    }

    return;
}


void mm_read_mtx_crd_size(std::ifstream &mtx_file, 
                          int64_t &M, int64_t &N, int64_t &nnz)
{
    //skip the comments
    std::string line;
    while (getline(mtx_file, line))
    {
        if (strstr(line.c_str(), "%") == NULL) break;        
    }

    //read the size
    std::stringstream ss(line);
    ss >> M >> N >> nnz;

    return;
}

void mm_read_crd_sym_data(std::ifstream &mtx_file, 
                          int64_t M, int64_t N, int64_t &nnz, 
                          std::vector<std::vector<int>> &idx, 
                          std::vector<std::vector<float>> &val, 
                          enum MM_DTYPE_CODE mtx_data_type)
{
    //fill in the data
    std::string line;
    int tmp_I, tmp_J;
    float tmp_val;
    int64_t nnz_counter = 0;
    while (getline(mtx_file, line))
    {
        std::stringstream ss(line);
        if (mm_is_pattern(mtx_data_type))
        {
            ss >> tmp_I >> tmp_J;
            idx[tmp_J - 1].push_back(tmp_I - 1);
            val[tmp_J - 1].push_back(1);
            nnz_counter++;
            
            if (tmp_I != tmp_J)
            {
                //fill in the symmetric
                idx[tmp_I - 1].push_back(tmp_J - 1);
                val[tmp_I - 1].push_back(1);
                nnz_counter++;
            }
        }
        else
        {
            ss >> tmp_I >> tmp_J >> tmp_val;
            if (tmp_val != 0)
            {
                idx[tmp_J - 1].push_back(tmp_I - 1);
                //val[tmp_J - 1].push_back(tmp_val);
                val[tmp_J - 1].push_back(1);
                nnz_counter++;

                if (tmp_I != tmp_J)
                {
                    //fill in the symmetric
                    idx[tmp_I - 1].push_back(tmp_J - 1);
                    //val[tmp_I - 1].push_back(tmp_val);
                    val[tmp_I - 1].push_back(1);
                    nnz_counter++;
                }
            }
        }
        ss.clear();
    }
    
    nnz = nnz_counter;

    return;
}

void mm_read_crd_general_data(std::ifstream &mtx_file, 
                              int64_t M, int64_t N, int64_t &nnz, 
                              std::vector<std::vector<int>> &idx, std::vector<std::vector<float>> &val, 
                              enum MM_DTYPE_CODE mtx_data_type)
{
    std::string line;
    int tmp_I, tmp_J;
    float tmp_val;
    int64_t nnz_counter = 0;
    while (getline(mtx_file, line))
    {
        std::stringstream ss(line);
        if (mm_is_pattern(mtx_data_type))
        {
            ss >> tmp_I >> tmp_J;
            idx[tmp_J - 1].push_back(tmp_I - 1);
            val[tmp_J - 1].push_back(1);        
            nnz_counter++;
        }
        else
        {
            ss >> tmp_I >> tmp_J >> tmp_val;
            if (tmp_val != 0)
            {
                idx[tmp_J - 1].push_back(tmp_I - 1);
                val[tmp_J - 1].push_back(tmp_val);
                //val[tmp_J - 1].push_back(1);
                nnz_counter++;
            }
        }
        ss.clear();
    }

    nnz = nnz_counter;
        
    return;
}

/*  high level I/O routines */

int mm_read_mtx_crd(std::ifstream &mtx_file, 
                    int64_t &M, int64_t &N, int64_t &nnz, 
					std::vector<std::vector<int>> &idx, 
					std::vector<std::vector<float>> &val, 
					enum MM_MTYPE_CODE &mtx_mat_type, 
					enum MM_DTYPE_CODE &mtx_data_type,
					enum MM_SYM_COMPRESS_FLAG mtx_sym_compress_flag)
{
    //get the matrix size
    mm_read_mtx_crd_size(mtx_file, M, N, nnz);

    idx.resize(N);
    val.resize(N);
     
    //read the data
    if (mm_is_symmetric(mtx_mat_type) && mtx_sym_compress_flag == MM_SYM_UNCOMPRESS)
    {
        nnz = nnz * 2;
        mm_read_crd_sym_data(mtx_file, M, N, nnz, idx, val, mtx_data_type);
        mtx_file.close();
    }
    else
    {
        if (mm_is_symmetric(mtx_mat_type) && mtx_sym_compress_flag == MM_SYM_COMPRESS)
        {
            mm_read_crd_general_data(mtx_file, M, N, nnz, idx, val, mtx_data_type);
            mtx_file.close();            
        }
        else
        {
            mm_read_crd_general_data(mtx_file, M, N, nnz, idx, val, mtx_data_type);             
        }         
    }

    return 0;
}

void mm_mtx_to_csr(std::vector<std::vector<int>> &mtx_idx,
                   std::vector<std::vector<float>> &mtx_val,
                   int* csr_offsets, int* csr_colid, 
                   float* csr_val, int64_t nnz)
{
    int64_t offset_counter = 0, csr_counter = 0;

    for (int64_t i = 0; i < mtx_idx.size(); i++)
    {
        csr_offsets[i] = offset_counter;
        offset_counter += mtx_idx[i].size();
        for (int64_t j = 0; j < mtx_idx[i].size(); j++)
        {
            csr_colid[csr_counter] = mtx_idx[i][j];
            csr_val[csr_counter] = mtx_val[i][j];
            csr_counter++;
        }         
             
    }
    csr_offsets[mtx_idx.size()] = nnz;
     
    //clear the original mtx
    std::vector<std::vector<int>>().swap(mtx_idx);
    std::vector<std::vector<float>>().swap(mtx_val);
    
    return;
}



