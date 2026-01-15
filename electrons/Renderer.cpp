#include "Renderer.hpp"
#include <algorithm>
#include <cstdio>

Renderer::Renderer(int windowW, int windowH, int fieldW, int fieldH)
    : m_windowW(windowW),
    m_windowH(windowH),
    m_fieldW(fieldW),
    m_fieldH(fieldH),
    m_points(sf::Points)
{
    if (!m_fieldTexture.create((unsigned)m_fieldW, (unsigned)m_fieldH)) {
        std::puts("ERROR: failed to create field texture.");
    }
    m_fieldSprite.setTexture(m_fieldTexture, true);
    setupFieldSpriteScale();

    m_points.clear();

    if (m_font.loadFromFile("arial.ttf")) {
        m_fontLoaded = true;
        m_text.setFont(m_font);
        m_text.setCharacterSize(14);
        m_text.setFillColor(sf::Color::White);
        m_text.setPosition(10.f, 10.f);
    }
    else {
        std::puts("Note: overlay font not found (arial.ttf). Overlay disabled.");
    }

    m_trailLines.setPrimitiveType(sf::Lines);
}

void Renderer::setupFieldSpriteScale() {
    float sx = (float)m_windowW / (float)m_fieldW;
    float sy = (float)m_windowH / (float)m_fieldH;
    m_fieldSprite.setScale(sx, sy);
}

void Renderer::updateFieldTexture(const std::vector<std::uint8_t>& fieldRGBA) {
    const std::size_t needed = (std::size_t)m_fieldW * (std::size_t)m_fieldH * 4;
    if (fieldRGBA.size() < needed) {
        std::puts("WARN: fieldRGBA too small in updateFieldTexture()");
        return;
    }

    m_fieldTexture.update(fieldRGBA.data());
}

void Renderer::updateParticles(const std::vector<float>& x, const std::vector<float>& y) {
    const std::size_t n = std::min(x.size(), y.size());
    m_points.resize(n);

    if (m_trailsEnabled) {
        if (m_trails.size() != n) {
            m_trails.assign(n, {});
        }
    }

    for (std::size_t i = 0; i < n; i++) {
        float px = std::clamp(x[i], 0.f, (float)(m_windowW - 1));
        float py = std::clamp(y[i], 0.f, (float)(m_windowH - 1));

        m_points[i].position = sf::Vector2f(px, py);
        m_points[i].color = sf::Color::Black;

        if (m_trailsEnabled) {
            auto& dq = m_trails[i];
            dq.push_back(sf::Vector2f(px, py));
            while ((int)dq.size() > m_trailLen) dq.pop_front();
        }
    }

    if (m_trailsEnabled) {
        m_trailLines.clear();
        m_trailLines.setPrimitiveType(sf::Lines);

        for (std::size_t i = 0; i < n; i++) {
            const auto& dq = m_trails[i];
            if (dq.size() < 2) continue;

            const float inv = 1.0f / (float)(dq.size() - 1);

            for (std::size_t k = 1; k < dq.size(); k++) {
                float t = (float)k * inv;
                sf::Uint8 alpha = (sf::Uint8)(40 + 215 * t);

                sf::Color col(255, 255, 255, alpha);

                m_trailLines.append(sf::Vertex(dq[k - 1], col));
                m_trailLines.append(sf::Vertex(dq[k], col));
            }
        }
    }
}

void Renderer::draw(sf::RenderWindow& window) {
    window.draw(m_fieldSprite);

    if (m_trailsEnabled) window.draw(m_trailLines);

    window.draw(m_points);
}

void Renderer::drawOverlay(sf::RenderWindow& window, bool paused, bool fieldEnabled, int fieldEveryNFrames) {
    if (!m_fontLoaded) return;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "Space: pause | F: field | R: reset | Esc: quit | O: trails\n"
        "paused: %s | field: %s | trails: %s | field every N frames: %d",
        paused ? "YES" : "NO",
        fieldEnabled ? "ON" : "OFF",
		m_trailsEnabled ? "ON" : "OFF",
        fieldEveryNFrames);

    m_text.setString(buf);
    window.draw(m_text);
}
