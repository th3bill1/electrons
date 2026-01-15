#pragma once
#include <cstdint>
#include <vector>

class Simulation {
public:
    Simulation(int windowW, int windowH, int fieldW, int fieldH, int particleCount);
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void reset();

    void step(float dt, bool doFieldThisFrame);

    void downloadFieldRGBA(std::vector<std::uint8_t>& outRGBA) const;

    void downloadParticlePositions(std::vector<float>& outX, std::vector<float>& outY) const;

    int windowW() const { return m_windowW; }
    int windowH() const { return m_windowH; }
    int fieldW() const { return m_fieldW; }
    int fieldH() const { return m_fieldH; }
    int particleCount() const { return m_N; }

private:
    int m_windowW{};
    int m_windowH{};
    int m_fieldW{};
    int m_fieldH{};
    int m_N{};

    float* m_d_px = nullptr;
    float* m_d_py = nullptr;
    float* m_d_vx = nullptr;
    float* m_d_vy = nullptr;
    float* m_d_q = nullptr;
    float* m_d_invMass = nullptr;

    float* m_d_V = nullptr;
    void* m_d_rgba = nullptr;

private:
    void allocateDevice();
    void freeDevice();
};
