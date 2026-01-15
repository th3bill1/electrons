#pragma once
#include <cuda_runtime.h>

__global__ void fieldKernel(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ q,
    float* __restrict__ V,
    uchar4* __restrict__ rgba,
	const float scale = 1.f
);

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
);
