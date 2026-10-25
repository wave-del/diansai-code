#include "preprocess.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    const auto geometry = ec26e::vision2::calculate_letterbox_geometry(764, 540, 640, 640);
    assert(geometry.resized_width == 640);
    assert(geometry.resized_height == 452);
    assert(geometry.left == 0);
    assert(geometry.top == 94);

    std::vector<std::uint8_t> bgr(2 * 3, 0);
    bgr[0] = 10;
    bgr[1] = 20;
    bgr[2] = 30;
    std::vector<std::uint8_t> rgb(4 * 4 * 3, 0);
    ec26e::vision2::LetterboxGeometry square{};
    assert(ec26e::vision2::letterbox_bgr_to_rgb(
        bgr.data(), bgr.size(), 2, 1, 6,
        rgb.data(), rgb.size(), 4, 4, 114, &square
    ));
    assert(square.top == 1 && square.resized_height == 2);
    const std::size_t first = (static_cast<std::size_t>(square.top) * 4) * 3;
    assert(rgb[first] == 30 && rgb[first + 1] == 20 && rgb[first + 2] == 10);
    assert(!ec26e::vision2::letterbox_bgr_to_rgb(
        bgr.data(), bgr.size() - 1, 2, 1, 6,
        rgb.data(), rgb.size(), 4, 4, 114, nullptr
    ));
    return 0;
}
