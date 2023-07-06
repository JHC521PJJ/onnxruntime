#ifndef __ONNXRUN_H__
#define __ONNXRUN_H__

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

class OnnxInferenceRunner {
public:
    using VecFloat = std::vector<float>;
    using VecInt64 = std::vector<int64_t>;

private:
    static Ort::Env env_;
    std::shared_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
public:
    OnnxInferenceRunner();
    ~OnnxInferenceRunner();
    OnnxInferenceRunner(const OnnxInferenceRunner& other) = delete;
    OnnxInferenceRunner(OnnxInferenceRunner&& other) = delete;
    OnnxInferenceRunner& operator=(const OnnxInferenceRunner& other) = delete;
    OnnxInferenceRunner& operator=(OnnxInferenceRunner&& other) = delete;

    void loadModel(const std::string& model_path);
    void setSessionNumThreads(const int num);
    void setSessionCUDA(const int device_id);
    void printModelInfo();
    VecFloat inference(VecFloat& input_vector);

    size_t getSessionInputCount();
    size_t getSessionOutputCount();
    VecInt64 getSessionInputNodeDims(size_t index);
    VecInt64 getSessionOutputNodeDims(size_t index);
    const char* getSessionInputName(size_t index);
    const char* getSessionOutputName(size_t index);
};

#endif