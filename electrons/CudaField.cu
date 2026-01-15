#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

#include "CudaKernels.cuh"

static inline void CUDA_CHECK(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::printf("CUDA ERROR: %s | %s\n", msg, cudaGetErrorString(err));
        std::abort();
    }
}

__device__ __forceinline__ float fastClamp01(float x) {
    return fminf(1.f, fmaxf(0.f, x));
}

__global__ void fieldKernel(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ q,
    float* __restrict__ V,
    uchar4* __restrict__ rgba,
	const float scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = fieldW * fieldH;
    if (tid >= totalPixels) return;

    int ix = tid % fieldW;
    int iy = tid / fieldW;

    float x = ((ix + 0.5f) / (float)fieldW) * (float)windowW * scale;
    float y = ((iy + 0.5f) / (float)fieldH) * (float)windowH * scale;

    extern __shared__ float sh[];
    float* shx = sh;
    float* shy = shx + blockDim.x;
    float* shq = shy + blockDim.x;

    float acc = 0.f;

    const float eps0 = 20.f;
    const float eps = eps0 * scale * scale;
    const float k = 1.0f / scale;

    for (int base = 0; base < N; base += blockDim.x) {
        int j = base + threadIdx.x;
        if (j < N) {
            shx[threadIdx.x] = px[j];
            shy[threadIdx.x] = py[j];
            shq[threadIdx.x] = q[j];
        }
        __syncthreads();

        int tileCount = min(blockDim.x, N - base);
        for (int t = 0; t < tileCount; t++) {
            float dx = x - shx[t] * scale;
            float dy = y - shy[t] * scale;
            float r2 = dx * dx + dy * dy + eps;
            acc += shq[t] * rsqrtf(r2);
        }
        __syncthreads();
    }

    float v = k * acc;
    V[tid] = v;

    float vNorm = v / (1.f + fabsf(v));

    float red = (vNorm > 0.f) ? vNorm : 0.f;
    float blue = (vNorm < 0.f) ? -vNorm : 0.f;

    red = sqrtf(fastClamp01(red));
    blue = sqrtf(fastClamp01(blue));

    unsigned char R = (unsigned char)(255.f * red);
    unsigned char B = (unsigned char)(255.f * blue);

    rgba[tid] = make_uchar4(R, 0, B, 255);
}

void computeFieldCUDA(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    const float* d_px, const float* d_py, const float* d_q,
    float* d_V,
    void* d_rgba_void,
	const float scale = 1.f
) {
    uchar4* d_rgba = reinterpret_cast<uchar4*>(d_rgba_void);

    int total = fieldW * fieldH;
    int block = 256;
    int grid = (total + block - 1) / block;

    size_t shBytes = (size_t)block * sizeof(float) * 3;

    fieldKernel << <grid, block, shBytes >> > (
        windowW, windowH, fieldW, fieldH,
        N,
        d_px, d_py, d_q,
        d_V,
        d_rgba,
        scale
        );

    CUDA_CHECK(cudaGetLastError(), "fieldKernel launch");
    CUDA_CHECK(cudaDeviceSynchronize(), "fieldKernel sync");
}
