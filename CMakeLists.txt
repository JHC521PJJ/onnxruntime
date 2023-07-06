cmake_minimum_required(VERSION 3.5)
project(testmain)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(SOURCE_FILES test.cpp)
set(INC_DIR /home/pjj/onnxruntime/onnxruntime-linux-x64-gpu-1.4.0/include)
set(LINK_DIR /home/pjj/onnxruntime/onnxruntime-linux-x64-gpu-1.4.0/lib/libonnxruntime.so)
set(CNPY_DIR /home/pjj/pythoncode/EfficientAD-main/deployment/cnpy/build/libcnpy.so)
file(GLOB_RECURSE CUDA_HEADERS  *.cuh)
file(GLOB CUDA_SOURCES *.cu)


find_package(CUDA 10.1 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()

message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -O2)
cuda_add_library(resultTransformate SHARED ${CUDA_HEADERS} ${CUDA_SOURCES})
set(CUDAEXTRA_LIBS ${CUDAEXTRA_LIBS} resultTransformate)

add_compile_options(-O2 -mavx -mfma)
add_executable(testmain main.cpp)
target_sources(testmain PUBLIC onnxInferenceRun.cpp onnxLog.cpp)

include_directories(${INC_DIR})     # 用${}引用变量
target_link_libraries(testmain ${LINK_DIR})
target_link_libraries(testmain ${CNPY_DIR})
target_link_libraries(testmain ${OpenCV_LIBS})
target_link_libraries(testmain ${CUDAEXTRA_LIBS})
set_property(TARGET testmain PROPERTY CXX_STANDARD 11)





