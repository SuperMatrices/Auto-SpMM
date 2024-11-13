
#include "./base.h"


void spmm_usage(char* prog)
{
    std::cerr << "Resource-Aware Automatic Hybrid Sparse Matrix Multiplication " << std::endl
    << prog << " [options] [data] " << std::endl
    << " options: Specify the settings " << std::endl
    << "   -c -- Column dimension of dense matrix" << std::endl
    << "   -h -- Help" << std::endl
    << " data: Specify the file (input/output)" << std::endl
    << "   -i -- The input mtx file name" << std::endl;  
    exit(1);
}

void parameter_parser(int argc, char* argv[], 
                      int64_t &dense_col_dim,
                      std::string &input_file)
{
    char ch;
    extern char* optarg;
    extern int optind, opterr;

    while ((ch = getopt(argc, argv, "g:c:ahd:i:t:")) != -1)
    {
        switch (ch)
        {
            case 'c':
            {
                dense_col_dim = atoi(optarg);
                break;
            }
            case 'h':
            {
                spmm_usage(argv[0]);
                break;
            }
            case 'i':
            {
                input_file = optarg;
                break;
            }
            default:
            {
                spmm_usage(argv[0]);
                break;
            }            
        }
    }
    return;
}