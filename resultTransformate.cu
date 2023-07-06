#include "resultTransformate.cuh"
#include <vector>
#include <cuda_runtime_api.h>
#include <stdio.h>

static constexpr int channel = 384;
static constexpr int out_size = 56;


__global__ void squareDifferenceKernel(float* d_teacher, float* d_student, float* d_autoencoder, 
    float* d_map_st, float* d_map_ae, const int size) {

    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size) {
        d_map_st[idx] = d_teacher[idx] - d_student[idx];
        d_map_ae[idx] = d_autoencoder[idx] - d_student[channel * 56 * 56 + idx];
        d_map_st[idx] *= d_map_st[idx]; 
        d_map_ae[idx] *= d_map_ae[idx]; 
    }
}

__global__ void meanKernel(float* d_map_st, float* d_map_ae, 
    float* d_mean_st, float* d_mean_ae, const int size) {
        
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float temp_st = 0.0f;
    float temp_ae = 0.0f;

    if(idx < size) {
        for(int i = 0; i < channel; ++i) {
            temp_st += d_map_st[idx + i * 56 * 56];
            temp_ae += d_map_ae[idx + i * 56 * 56];
        }
        d_mean_st[idx] = temp_st / channel;
        d_mean_ae[idx] = temp_ae / channel;
    }
}

__global__ void combineKernel(float* d_mean_st, float* d_mean_ae, float* d_combine,
    float* d_st_start_quantiles,
    float* d_st_end_quantiles,
    float* d_ae_start_quantiles,
    float* d_ae_end_quantiles,
    const int size) {
    
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size) {
        d_mean_st[idx] = 0.1f * (d_mean_st[idx] - *d_st_start_quantiles) / (*d_st_end_quantiles - *d_st_start_quantiles);
        d_mean_ae[idx] = 0.1f * (d_mean_ae[idx] - *d_ae_start_quantiles) / (*d_ae_end_quantiles - *d_ae_start_quantiles);
        d_combine[idx] = 0.5f * d_mean_st[idx] + 0.5f * d_mean_ae[idx];
    }
}

void resultTransformate(const std::vector<float>& t_output, 
    const std::vector<float>& s_output,
    const std::vector<float>& ae_output,
    float q_st_start_quantiles,
    float q_st_end_quantiles,
    float q_ae_start_quantiles,
    float q_ae_end_quantiles,
    const int device_id,
    std::vector<float>& vec_combine) {

    cudaSetDevice(device_id);

    float* d_teacher; float* d_student; float* d_autoencoder;
    float* d_map_st; float* d_map_ae;
    float* d_mean_st; float* d_mean_ae;
    float* d_combine;
    float* d_st_start_quantiles;
    float* d_st_end_quantiles;
    float* d_ae_start_quantiles;
    float* d_ae_end_quantiles; 

    cudaMalloc((void **) &d_teacher, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_student, sizeof(float) * channel * 2 * out_size * out_size);
    cudaMalloc((void **) &d_autoencoder, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_map_st, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_map_ae, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_mean_st, sizeof(float) * out_size * out_size);
    cudaMalloc((void **) &d_mean_ae, sizeof(float) * out_size * out_size);
    cudaMalloc((void **) &d_combine, sizeof(float) * out_size * out_size);
    cudaMalloc((void **) &d_st_start_quantiles, sizeof(float));
    cudaMalloc((void **) &d_st_end_quantiles, sizeof(float));
    cudaMalloc((void **) &d_ae_start_quantiles, sizeof(float));
    cudaMalloc((void **) &d_ae_end_quantiles, sizeof(float));

    cudaMemcpy(d_teacher, t_output.data(), sizeof(float) * channel * out_size * out_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_student, s_output.data(), sizeof(float) * channel * 2 *  out_size * out_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_autoencoder, ae_output.data(), sizeof(float) * channel * out_size * out_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_start_quantiles, &q_st_start_quantiles, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_end_quantiles, &q_st_end_quantiles, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ae_start_quantiles, &q_ae_start_quantiles, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ae_end_quantiles, &q_ae_end_quantiles, sizeof(float), cudaMemcpyHostToDevice);

    int size = channel * out_size * out_size;
    unsigned int block_size = 16 * 16;
    unsigned int grid_size = (size + block_size - 1) / block_size;
    dim3 grid_dim(grid_size);
    dim3 block_dim(block_size);
    squareDifferenceKernel<<<grid_dim, block_dim>>>(d_teacher, d_student, d_autoencoder, d_map_st, d_map_ae, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    size = out_size * out_size;
    grid_size = (size + block_size - 1) / block_size;
    meanKernel<<<grid_size, block_dim>>>(d_map_st, d_map_ae, d_mean_st, d_mean_ae, size);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    combineKernel<<<grid_size, block_dim>>>(d_mean_st, d_mean_ae, d_combine, d_st_start_quantiles, d_st_end_quantiles, d_ae_start_quantiles, d_ae_end_quantiles, size);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(vec_combine.data(), d_combine, sizeof(float) * out_size * out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_teacher);
    cudaFree(d_student);
    cudaFree(d_autoencoder);
    cudaFree(d_map_st);
    cudaFree(d_map_ae);
    cudaFree(d_mean_st);
    cudaFree(d_mean_ae);
    cudaFree(d_combine);
    cudaFree(d_st_start_quantiles);
    cudaFree(d_st_end_quantiles);
    cudaFree(d_ae_start_quantiles);
    cudaFree(d_ae_end_quantiles); 
}