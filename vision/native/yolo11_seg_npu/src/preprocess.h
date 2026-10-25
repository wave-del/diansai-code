#ifndef EC26E_YOLO11_SEG_PREPROCESS_H
#define EC26E_YOLO11_SEG_PREPROCESS_H

#include <cstddef>
#include <cstdint>

namespace ec26e::vision2 {

struct LetterboxGeometry {
    double scale;
    int resized_width;
    int resized_height;
    int left;
    int top;
};

LetterboxGeometry calculate_letterbox_geometry(
    int source_width,
    int source_height,
    int destination_width,
    int destination_height
);

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
);

}  // namespace ec26e::vision2

#endif
