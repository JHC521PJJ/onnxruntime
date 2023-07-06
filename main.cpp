#include "imagePreprocess.h"
#include "npyToVector.h"
#include "onnxInferenceRun.h"
#include "time_count.h"
#include "resultTransformate.h"
#include "resultTransformate.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>




int main(){
    const std::string t_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip2/teacher_final.onnx";
    const std::string s_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip2/student_final.onnx";
    const std::string ae_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip2/autoencoder_final.onnx";

    OnnxInferenceRunner t_onnxrunner;
    OnnxInferenceRunner s_onnxrunner;
    OnnxInferenceRunner ae_onnxrunner;
    t_onnxrunner.setSessionCUDA(8);
    s_onnxrunner.setSessionCUDA(8);
    ae_onnxrunner.setSessionCUDA(8);

    TimeCount::instance().start();
    t_onnxrunner.loadModel(t_model_path);
    s_onnxrunner.loadModel(s_model_path);
    ae_onnxrunner.loadModel(ae_model_path);
    auto load_time_count = TimeCount::instance().getTime();
    std::cout<<"Load model takes time: " << load_time_count << "ms" << "\n";

    t_onnxrunner.printModelInfo();
    s_onnxrunner.printModelInfo();
    ae_onnxrunner.printModelInfo();
    
    const std::string file_path = "/data2/mvtec_loco/chip2/test/structural_anomalies/";
    std::string img_path = file_path + "*.bmp";
    std::vector<cv::String> vec_file{};
    cv::glob(img_path, vec_file);
    std::cout<<"File size: "<< vec_file.size() << "\n";

    std::vector<float> vec_teacher_mean = npyToVector("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/t_mean_quantiles.npy");
    std::vector<float> vec_teacher_std = npyToVector("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/t_std_quantiles.npy");
    float q_st_start_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/q_st_start_quantiles.npy");
    float q_st_end_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/q_st_end_quantiles.npy");
    float q_ae_start_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/q_ae_start_quantiles.npy");
    float q_ae_end_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip2/q_ae_end_quantiles.npy");

    std::vector<double> vec_time_avg{};
    for(int i = 0; i < vec_file.size(); ++i) {
        cv::Mat image = cv::imread(vec_file[i]);
        std::cout << vec_file[i] << "\n";

        TimeCount::instance().start();
        std::vector<float> input_vector = imagePreprocessing(image);
        auto preprocessing_time_count = TimeCount::instance().getTime();
        std::cout<< "Image preprocessing takes time: " << preprocessing_time_count << "ms" << "\n";

        std::vector<float> t_onnxrun_output = t_onnxrunner.inference(input_vector);
        std::vector<float> s_onnxrun_output = s_onnxrunner.inference(input_vector);
        std::vector<float> ae_onnxrun_output = ae_onnxrunner.inference(input_vector);
        auto infer_time_count = TimeCount::instance().getTime();
        std::cout<< "Model inference takes time: " << infer_time_count << "ms" << "\n";
  
        for(int c = 0; c < 384; ++c) {
            for(int i = 0; i < 56 * 56; ++i) {
                t_onnxrun_output[i + c * 56 * 56] = (t_onnxrun_output[i + c * 56 * 56] - vec_teacher_mean[c]) / vec_teacher_std[c]; 
            }
        }

        // std::vector<float> vec_combined(56 * 56);
        // resultTransformate(t_onnxrun_output, s_onnxrun_output, ae_onnxrun_output, q_st_start_quantiles, q_st_end_quantiles, q_ae_start_quantiles, q_ae_end_quantiles, 9, vec_combined);
        // auto combined_time_count = TimeCount::instance().getTime();
        // std::cout<< "Combine operation takes time: " << combined_time_count << "ms" << "\n";

        auto vec_mean = meanOperation(t_onnxrun_output, s_onnxrun_output, ae_onnxrun_output);
        std::vector<float> vec_mean_st = vec_mean[0];
        std::vector<float> vec_mean_ae = vec_mean[1];
        auto mean_time_count = TimeCount::instance().getTime();
        std::cout<< "Mean operation takes time: " << mean_time_count << "ms" << "\n";

        std::vector<float> vec_combined = combineOperation(vec_mean_st, vec_mean_ae, q_st_start_quantiles, q_st_end_quantiles, q_ae_start_quantiles, q_ae_end_quantiles);
        auto combined_time_count = TimeCount::instance().getTime();
        std::cout<< "Combine operation takes time: " << combined_time_count << "ms" << "\n";

        auto it_ad_score = std::max_element(vec_combined.begin(), vec_combined.end());
        std::cout<< "Score: " << *it_ad_score << " ";
        auto inference_time_count = TimeCount::instance().getTime();
        std::cout<< "All takes time: " << inference_time_count << "ms" << "\n";
        vec_time_avg.emplace_back(inference_time_count);
    }

    double time_avg = vectorAverage(vec_time_avg);
    std::cout<< "Avg time: " << time_avg << "\n";
    
    return 0;
}
