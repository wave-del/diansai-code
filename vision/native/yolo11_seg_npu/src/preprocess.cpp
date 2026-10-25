#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ec26e::vision2 {
namespace {

constexpr int kChannels = 3;

void fill_padding(
    std::uint8_t *destination,
    int destination_width,
    int destination_height,
    const LetterboxGeometry &geometry,
    std::uint8_t value
) {
    const std::size_t bytes =
        static_cast<std::size_t>(destination_width) * destination_height * kChannels;
    std::memset(destination, value, bytes);
    (void)geometry;
}

void resize_convert(
    const std::uint8_t *source,
    std::size_t source_stride,
    int source_width,
    int source_height,
    std::uint8_t *destination,
    int destination_width,
    const LetterboxGeometry &geometry
) {
    const std::size_t destination_stride =
        static_cast<std::size_t>(destination_width) * kChannels;
    const double scale_x =
        static_cast<double>(source_width) / geometry.resized_width;
    const double scale_y =
        static_cast<double>(source_height) / geometry.resized_height;
    for (int y = 0; y < geometry.resized_height; ++y) {
        const double fy = (y + 0.5) * scale_y - 0.5;
        const int floor_y = static_cast<int>(std::floor(fy));
        const int y0 = std::clamp(floor_y, 0, source_height - 1);
        const int y1 = std::clamp(floor_y + 1, 0, source_height - 1);
        const double wy = std::clamp(fy - floor_y, 0.0, 1.0);
        const std::uint8_t *row0 = source + static_cast<std::size_t>(y0) * source_stride;
        const std::uint8_t *row1 = source + static_cast<std::size_t>(y1) * source_stride;
        std::uint8_t *out = destination +
            static_cast<std::size_t>(geometry.top + y) * destination_stride +
            static_cast<std::size_t>(geometry.left) * kChannels;
        for (int x = 0; x < geometry.resized_width; ++x) {
            const double fx = (x + 0.5) * scale_x - 0.5;
            const int floor_x = static_cast<int>(std::floor(fx));
            const int x0 = std::clamp(floor_x, 0, source_width - 1);
            const int x1 = std::clamp(floor_x + 1, 0, source_width - 1);
            const double wx = std::clamp(fx - floor_x, 0.0, 1.0);
            for (int rgb = 0; rgb < kChannels; ++rgb) {
                const int bgr = 2 - rgb;
                const double top = row0[x0 * kChannels + bgr] * (1.0 - wx) +
                    row0[x1 * kChannels + bgr] * wx;
                const double bottom = row1[x0 * kChannels + bgr] * (1.0 - wx) +
                    row1[x1 * kChannels + bgr] * wx;
                out[x * kChannels + rgb] = static_cast<std::uint8_t>(
                    std::clamp(std::nearbyint(top * (1.0 - wy) + bottom * wy), 0.0, 255.0)
                );
            }
        }
    }
}

}  // namespace

LetterboxGeometry calculate_letterbox_geometry(
    int source_width,
    int source_height,
    int destination_width,
    int destination_height
) {
    const double scale = std::min(
        static_cast<double>(destination_width) / source_width,
        static_cast<double>(destination_height) / source_height
    );
    const int width = static_cast<int>(std::nearbyint(source_width * scale));
    const int height = static_cast<int>(std::nearbyint(source_height * scale));
    return {
        scale,
        width,
        height,
        (destination_width - width) / 2,
        (destination_height - height) / 2,
    };
}

bool letterbox_bgr_to_rgb(
    const std::uint8_t *source,
    std::size_t source_size,
    int source_width,
    int source_height,
    std::size_t source_stride,
    std::uint8_t *destination,
    std::size_t destination_size,
    int destination_width,
    int destination_height,
    std::uint8_t padding_value,
    LetterboxGeometry *geometry
) {
    if (source == nullptr || destination == nullptr || source_width <= 0 ||
        source_height <= 0 || destination_width <= 0 || destination_height <= 0) {
        return false;
    }
    const std::size_t source_row = static_cast<std::size_t>(source_width) * kChannels;
    const std::size_t required_source =
        static_cast<std::size_t>(source_height - 1) * source_stride + source_row;
    const std::size_t required_destination =
        static_cast<std::size_t>(destination_width) * destination_height * kChannels;
    if (source_stride < source_row || source_size < required_source ||
        destination_size < required_destination) {
        return false;
    }
    const LetterboxGeometry calculated = calculate_letterbox_geometry(
        source_width, source_height, destination_width, destination_height
    );
    if (calculated.resized_width <= 0 || calculated.resized_height <= 0) {
        return false;
    }
    fill_padding(
        destination, destination_width, destination_height, calculated, padding_value
    );
    resize_convert(
        source,
        source_stride,
        source_width,
        source_height,
        destination,
        destination_width,
        calculated
    );
    if (geometry != nullptr) {
        *geometry = calculated;
    }
    return true;
}

}  // namespace ec26e::vision2
