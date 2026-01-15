#pragma once
#include <SFML/Graphics.hpp>
#include <cstdint>
#include <vector>
#include <deque>

struct TrailPoint {
    float x, y;
};

class Renderer {
public:
    Renderer(int windowW, int windowH, int fieldW, int fieldH);

    void updateFieldTexture(const std::vector<std::uint8_t>& fieldRGBA);

    void updateParticles(const std::vector<float>& x, const std::vector<float>& y);

    void draw(sf::RenderWindow& window);

    void drawOverlay(sf::RenderWindow& window, bool paused, bool fieldEnabled, int fieldEveryNFrames);

    void enableTrails(bool on) { m_trailsEnabled = on; }
    void setTrailLength(int frames) { m_trailLen = std::max(1, frames); }

private:
    int m_windowW{};
    int m_windowH{};
    int m_fieldW{};
    int m_fieldH{};

    sf::Texture m_fieldTexture;
    sf::Sprite  m_fieldSprite;

    sf::VertexArray m_points;

    sf::Font m_font;
    sf::Text m_text;
    bool m_fontLoaded = false;

    bool m_trailsEnabled = false;
    int  m_trailLen = 20;

    std::vector<std::deque<sf::Vector2f>> m_trails; 
    sf::VertexArray m_trailLines;

private:
    void setupFieldSpriteScale();
};
