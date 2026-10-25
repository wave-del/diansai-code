#ifndef EC26E_YOLO11_SEG_POSTPROCESS_H
#define EC26E_YOLO11_SEG_POSTPROCESS_H

#include "preprocess.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ec26e::vision2 {

constexpr int kScaleCount = 3;
constexpr int kMaskChannels = 32;
constexpr int kSemanticOutputCount = 10;

struct SegProposal {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    std::array<float, kMaskChannels> coefficients;
};

std::vector<SegProposal> decode_proposals(
    const std::array<const float *, kSemanticOutputCount> &outputs,
    int input_size,
    float score_threshold
);

std::vector<SegProposal> non_maximum_suppression(
    std::vector<SegProposal> proposals,
    float nms_threshold,
    int max_detections
);

void decode_mask_to_source(
    const SegProposal &proposal,
    const float *prototype,
    int prototype_size,
    const LetterboxGeometry &geometry,
    int source_width,
    int source_height,
    float mask_threshold,
    std::uint8_t *destination
);

}  // namespace ec26e::vision2

#endif
