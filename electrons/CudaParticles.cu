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

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void particleKernel(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    float dt,
    const float* __restrict__ V,
    float* __restrict__ px,
    float* __restrict__ py,
    float* __restrict__ vx,
    float* __restrict__ vy,
    const float* __restrict__ q,
    const float* __restrict__ invMass
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = px[i];
    float y = py[i];
    float ux = vx[i];
    float uy = vy[i];

    float fx = (x / (float)windowW) * (float)fieldW;
    float fy = (y / (float)windowH) * (float)fieldH;

    int ix = (int)fx;
    int iy = (int)fy;

    ix = clampi(ix, 1, fieldW - 2);
    iy = clampi(iy, 1, fieldH - 2);

    int idx = iy * fieldW + ix;

    float Vx1 = V[idx + 1];
    float Vx0 = V[idx - 1];
    float Vy1 = V[idx + fieldW];
    float Vy0 = V[idx - fieldW];

    float dVdx_grid = (Vx1 - Vx0) * 0.5f;
    float dVdy_grid = (Vy1 - Vy0) * 0.5f;

    float dx_world = (float)windowW / (float)fieldW;
    float dy_world = (float)windowH / (float)fieldH;

    float dVdx = dVdx_grid / dx_world;
    float dVdy = dVdy_grid / dy_world;

    float Ex = -dVdx;
    float Ey = -dVdy;

    const float forceScale = 1000.f;
    float aScale = q[i] * invMass[i] * forceScale;

    float ax = Ex * aScale;
    float ay = Ey * aScale;

    ux += ax * dt;
    uy += ay * dt;

    float v2 = ux * ux + uy * uy;
    const float vmax = 120.f;
    if (v2 > vmax * vmax) {
        float inv = rsqrtf(v2);
        ux *= vmax * inv;
        uy *= vmax * inv;
    }

    const float damp = 0.9999f;
    ux *= damp;
    uy *= damp;

    x += ux * dt;
    y += uy * dt;

    const float minX = 0.f;
    const float minY = 0.f;
    const float maxX = (float)(windowW - 1);
    const float maxY = (float)(windowH - 1);

    if (x < minX) { x = minX; ux = -ux; }
    if (x > maxX) { x = maxX; ux = -ux; }
    if (y < minY) { y = minY; uy = -uy; }
    if (y > maxY) { y = maxY; uy = -uy; }

    px[i] = x;
    py[i] = y;
    vx[i] = ux;
    vy[i] = uy;
}

void updateParticlesCUDA(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    float dt,
    const float* d_V,
    float* d_px, float* d_py,
    float* d_vx, float* d_vy,
    const float* d_q,
    const float* d_invMass
) {
    int block = 256;
    int grid = (N + block - 1) / block;

    particleKernel << <grid, block >> > (
        windowW, windowH,
        fieldW, fieldH,
        N,
        dt,
        d_V,
        d_px, d_py,
        d_vx, d_vy,
        d_q,
        d_invMass
        );

    CUDA_CHECK(cudaGetLastError(), "particleKernel launch");
    CUDA_CHECK(cudaDeviceSynchronize(), "particleKernel sync");
}
