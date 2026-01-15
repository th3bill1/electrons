#include "Simulation.hpp"

#include <random>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>

void computeFieldCUDA(
    int windowW, int windowH,
    int fieldW, int fieldH,
    int N,
    const float* d_px, const float* d_py, const float* d_q,
    float* d_V,
    void* d_rgba,
	const float scale = 1.f
);

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
);

static void CUDA_CHECK(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR: %s | %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

Simulation::Simulation(int windowW, int windowH, int fieldW, int fieldH, int particleCount)
    : m_windowW(windowW),
    m_windowH(windowH),
    m_fieldW(fieldW),
    m_fieldH(fieldH),
    m_N(particleCount)
{
    allocateDevice();
    reset();
}

Simulation::~Simulation() {
    freeDevice();
}

void Simulation::allocateDevice() {
    if (m_N <= 0) throw std::runtime_error("Particle count must be > 0");

    const size_t nBytes = (size_t)m_N * sizeof(float);
    const size_t fieldCount = (size_t)m_fieldW * (size_t)m_fieldH;
    const size_t fieldBytes = fieldCount * sizeof(float);
    const size_t rgbaBytes = fieldCount * 4;

    CUDA_CHECK(cudaMalloc(&m_d_px, nBytes), "cudaMalloc d_px");
    CUDA_CHECK(cudaMalloc(&m_d_py, nBytes), "cudaMalloc d_py");
    CUDA_CHECK(cudaMalloc(&m_d_vx, nBytes), "cudaMalloc d_vx");
    CUDA_CHECK(cudaMalloc(&m_d_vy, nBytes), "cudaMalloc d_vy");
    CUDA_CHECK(cudaMalloc(&m_d_q, nBytes), "cudaMalloc d_q");
    CUDA_CHECK(cudaMalloc(&m_d_invMass, nBytes), "cudaMalloc d_invMass");

    CUDA_CHECK(cudaMalloc(&m_d_V, fieldBytes), "cudaMalloc d_V");
    CUDA_CHECK(cudaMalloc(&m_d_rgba, rgbaBytes), "cudaMalloc d_rgba");

    CUDA_CHECK(cudaMemset(m_d_V, 0, fieldBytes), "cudaMemset d_V");
    CUDA_CHECK(cudaMemset(m_d_rgba, 0, rgbaBytes), "cudaMemset d_rgba");
}

void Simulation::freeDevice() {
    cudaFree(m_d_px); m_d_px = nullptr;
    cudaFree(m_d_py); m_d_py = nullptr;
    cudaFree(m_d_vx); m_d_vx = nullptr;
    cudaFree(m_d_vy); m_d_vy = nullptr;
    cudaFree(m_d_q);  m_d_q = nullptr;
    cudaFree(m_d_invMass); m_d_invMass = nullptr;

    cudaFree(m_d_V); m_d_V = nullptr;
    cudaFree(m_d_rgba); m_d_rgba = nullptr;
}

void Simulation::reset() {
    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> rx(0.f, (float)m_windowW);
    std::uniform_real_distribution<float> ry(0.f, (float)m_windowH);
    std::uniform_real_distribution<float> rv(-60.f, 60.f);
    std::uniform_int_distribution<int> r01(0, 1);

    std::vector<float> h_px(m_N), h_py(m_N), h_vx(m_N), h_vy(m_N), h_q(m_N), h_invMass(m_N);

    for (int i = 0; i < m_N; i++) {
        h_px[i] = rx(rng);
        h_py[i] = ry(rng);
        h_vx[i] = rv(rng);
        h_vy[i] = rv(rng);

        const bool isProton = (/*r01(rng) == 1*/ i % 2 == 0);
        h_q[i] = isProton ? +1.f : -1.f;

        h_invMass[i] = /*isProton ? (1.f / 20.f) :*/ 1.f;
    }

    const size_t nBytes = (size_t)m_N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(m_d_px, h_px.data(), nBytes, cudaMemcpyHostToDevice), "copy px");
    CUDA_CHECK(cudaMemcpy(m_d_py, h_py.data(), nBytes, cudaMemcpyHostToDevice), "copy py");
    CUDA_CHECK(cudaMemcpy(m_d_vx, h_vx.data(), nBytes, cudaMemcpyHostToDevice), "copy vx");
    CUDA_CHECK(cudaMemcpy(m_d_vy, h_vy.data(), nBytes, cudaMemcpyHostToDevice), "copy vy");
    CUDA_CHECK(cudaMemcpy(m_d_q, h_q.data(), nBytes, cudaMemcpyHostToDevice), "copy q");
    CUDA_CHECK(cudaMemcpy(m_d_invMass, h_invMass.data(), nBytes, cudaMemcpyHostToDevice), "copy invMass");

    computeFieldCUDA(m_windowW, m_windowH, m_fieldW, m_fieldH, m_N, m_d_px, m_d_py, m_d_q, m_d_V, m_d_rgba);
}

void Simulation::step(float dt, bool doFieldThisFrame) {
    if (dt <= 0.f) return;

    if (doFieldThisFrame) {
        computeFieldCUDA(
            m_windowW, m_windowH,
            m_fieldW, m_fieldH,
            m_N,
            m_d_px, m_d_py, m_d_q,
            m_d_V,
            m_d_rgba,
            0.2f
        );
    }

    updateParticlesCUDA(
        m_windowW, m_windowH,
        m_fieldW, m_fieldH,
        m_N,
        dt,
        m_d_V,
        m_d_px, m_d_py,
        m_d_vx, m_d_vy,
        m_d_q,
        m_d_invMass
    );
}

void Simulation::downloadFieldRGBA(std::vector<std::uint8_t>& outRGBA) const {
    const size_t count = (size_t)m_fieldW * (size_t)m_fieldH;
    outRGBA.resize(count * 4);

    CUDA_CHECK(
        cudaMemcpy(outRGBA.data(), m_d_rgba, count * 4, cudaMemcpyDeviceToHost),
        "downloadFieldRGBA cudaMemcpy"
    );
}

void Simulation::downloadParticlePositions(std::vector<float>& outX, std::vector<float>& outY) const {
    outX.resize((size_t)m_N);
    outY.resize((size_t)m_N);

    const size_t nBytes = (size_t)m_N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(outX.data(), m_d_px, nBytes, cudaMemcpyDeviceToHost), "download px");
    CUDA_CHECK(cudaMemcpy(outY.data(), m_d_py, nBytes, cudaMemcpyDeviceToHost), "download py");
}
