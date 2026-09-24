#pragma once
// snapjudge plotting canvas: dual backend — SVG markup AND raster PNG
// (via vendored lodepng). Text uses an embedded 5x7 bitmap font.

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace snapjudge { namespace plots {

struct Color {
  uint8_t r, g, b, a;
  static Color hex(uint32_t h, uint8_t alpha = 255) {
    return {static_cast<uint8_t>((h >> 16) & 0xFF), static_cast<uint8_t>((h >> 8) & 0xFF),
            static_cast<uint8_t>(h & 0xFF), alpha};
  }
};

class Canvas {
 public:
  int W, H;
  std::vector<uint8_t> px;  // RGBA

  Canvas(int w, int h) : W(w), H(h), px((size_t)w * h * 4, 255) {}
  // opaque white background
  void clear() { std::fill(px.begin(), px.end(), 255); }

  void set(int x, int y, Color c) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    size_t o = ((size_t)y * W + x) * 4;
    px[o] = c.r; px[o + 1] = c.g; px[o + 2] = c.b; px[o + 3] = c.a;
  }
  // alpha-blended set
  void blend(int x, int y, Color c) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    size_t o = ((size_t)y * W + x) * 4;
    float fa = c.a / 255.f;
    px[o] = static_cast<uint8_t>(px[o] * (1 - fa) + c.r * fa);
    px[o + 1] = static_cast<uint8_t>(px[o + 1] * (1 - fa) + c.g * fa);
    px[o + 2] = static_cast<uint8_t>(px[o + 2] * (1 - fa) + c.b * fa);
    px[o + 3] = 255;
  }
  void fill_rect(int x0, int y0, int w, int h, Color c) {
    for (int y = y0; y < y0 + h; ++y)
      for (int x = x0; x < x0 + w; ++x) blend(x, y, c);
  }
  void draw_line(int x0, int y0, int x1, int y1, Color c, int thick = 1) {
    // Bresenham
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
      for (int dx2 = -thick / 2; dx2 <= thick / 2; ++dx2)
        for (int dy2 = -thick / 2; dy2 <= thick / 2; ++dy2)
          blend(x0 + dx2, y0 + dy2, c);
      if (x0 == x1 && y0 == y1) break;
      int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }
  void draw_dashed_line(int x0, int y0, int x1, int y1, Color c, int thick = 1,
                        int dash = 6, int gap = 4) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int dist = 0;
    while (true) {
      if ((dist / (dash + gap)) % 2 == 0)
        for (int dx2 = -thick / 2; dx2 <= thick / 2; ++dx2)
          for (int dy2 = -thick / 2; dy2 <= thick / 2; ++dy2)
            blend(x0 + dx2, y0 + dy2, c);
      if (x0 == x1 && y0 == y1) break;
      int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
      ++dist;
    }
  }

  // Embedded 5x7 ASCII font; unknown chars -> blank. scale >1 enlarges.
  void draw_text(int x, int y, const std::string& t, Color c, int scale = 1,
                 const char* anchor = "start") {
    int w = (int)t.size() * 6 * scale - scale;
    if (std::string(anchor) == "middle") x -= w / 2;
    else if (std::string(anchor) == "end") x -= w;
    for (size_t i = 0; i < t.size(); ++i)
      draw_char(x + (int)(i * 6 * scale), y, t[i], c, scale);
  }

  bool write_png(const std::string& path);

 private:
  void draw_char(int x, int y, char ch, Color c, int scale);
};

}}  // namespace snapjudge::plots
