#include <vector>


extern "C" void resultTransformate(const std::vector<float>& t_output, 
    const std::vector<float>& s_output,
    const std::vector<float>& ae_output,
    float q_st_start_quantiles,
    float q_st_end_quantiles,
    float q_ae_start_quantiles,
    float q_ae_end_quantiles,
    const int device_id,
    std::vector<float>& vec_combine);