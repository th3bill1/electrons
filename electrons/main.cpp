#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "Renderer.hpp"
#include "Simulation.hpp"

static void printControls() {
    std::puts("Controls:");
    std::puts("  Space  - pause/resume");
    std::puts("  F      - toggle field recompute");
    std::puts("  R      - reset simulation");
    std::puts("  Esc    - quit");
	std::puts("  O      - enable/disable trail");
}

int main() {
    printControls();

    const int windowW = 1280;
    const int windowH = 720;

    const int fieldW = 480;
    const int fieldH = 270;

    const int particleCount = 10;

    sf::RenderWindow window(sf::VideoMode(windowW, windowH), "Electrostatic Field (CUDA + SFML)");
    window.setVerticalSyncEnabled(false);
    window.setFramerateLimit(0);

    Simulation sim(windowW, windowH, fieldW, fieldH, particleCount);

    Renderer renderer(windowW, windowH, fieldW, fieldH);

	renderer.setTrailLength(600);

    bool paused = false;
	bool trailesEnabled = false;
    bool recomputeField = true;
    int fieldEveryNFrames = 2;
    int frameCounter = 0;

    sf::Clock clock;
    sf::Clock fpsClock;
    int fpsFrames = 0;

    std::vector<uint8_t> fieldRGBA;
    std::vector<float> px, py;

    while (window.isOpen()) {
        sf::Event e{};
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) window.close();

            if (e.type == sf::Event::KeyPressed) {
                if (e.key.code == sf::Keyboard::Escape) window.close();

                if (e.key.code == sf::Keyboard::Space) paused = !paused;

                if (e.key.code == sf::Keyboard::F) recomputeField = !recomputeField;

                if (e.key.code == sf::Keyboard::R) {
                    sim.reset();
                    frameCounter = 0;
                }
                if (e.key.code == sf::Keyboard::O) {
					trailesEnabled = !trailesEnabled;
					renderer.enableTrails(trailesEnabled);
                }
            }
        }

        float dt = clock.restart().asSeconds();
        dt = std::min(dt, 1.0f / 15.0f);

        bool doFieldThisFrame = recomputeField && ((frameCounter % fieldEveryNFrames) == 0);

        if (!paused) {
            sim.step(dt, doFieldThisFrame);
            frameCounter++;
        }

        if (doFieldThisFrame) {
            sim.downloadFieldRGBA(fieldRGBA);
            renderer.updateFieldTexture(fieldRGBA);
        }

        sim.downloadParticlePositions(px, py);
        renderer.updateParticles(px, py);

        window.clear(sf::Color(20, 20, 20));
        renderer.draw(window);

        renderer.drawOverlay(window, paused, recomputeField, fieldEveryNFrames);

        window.display();

        fpsFrames++;
        if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
            float secs = fpsClock.restart().asSeconds();
            float fps = fpsFrames / secs;
            fpsFrames = 0;
            std::printf("FPS: %.1f | paused=%d | field=%d (every %d frames)\n",
                fps, paused ? 1 : 0, recomputeField ? 1 : 0, fieldEveryNFrames);
        }
    }

    return 0;
}
