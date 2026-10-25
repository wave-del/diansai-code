#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ec26e::vision2 {
namespace {

constexpr int kRegMax = 16;
constexpr std::array<int, kScaleCount> kStrides{8, 16, 32};

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float exponential = std::exp(-value);
        return 1.0F / (1.0F + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float dfl(const float *data, int grid_size, int index, int side) {
    const int offset = side * kRegMax;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int bin = 0; bin < kRegMax; ++bin) {
        maximum = std::max(maximum, data[(offset + bin) * grid_size + index]);
    }
    float denominator = 0.0F;
    float numerator = 0.0F;
    for (int bin = 0; bin < kRegMax; ++bin) {
        const float probability =
            std::exp(data[(offset + bin) * grid_size + index] - maximum);
        denominator += probability;
        numerator += bin * probability;
    }
    return denominator > 0.0F ? numerator / denominator : 0.0F;
}

float iou(const SegProposal &left, const SegProposal &right) {
    const float x1 = std::max(left.x1, right.x1);
    const float y1 = std::max(left.y1, right.y1);
    const float x2 = std::min(left.x2, right.x2);
    const float y2 = std::min(left.y2, right.y2);
    const float intersection =
        std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float left_area =
        std::max(0.0F, left.x2 - left.x1) * std::max(0.0F, left.y2 - left.y1);
    const float right_area =
        std::max(0.0F, right.x2 - right.x1) * std::max(0.0F, right.y2 - right.y1);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

float bilinear(
    const std::vector<float> &values,
    int size,
    float x,
    float y
) {
    x = std::clamp(x, 0.0F, static_cast<float>(size - 1));
    y = std::clamp(y, 0.0F, static_cast<float>(size - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, size - 1);
    const int y1 = std::min(y0 + 1, size - 1);
    const float wx = x - x0;
    const float wy = y - y0;
    const float top = values[y0 * size + x0] * (1.0F - wx) +
        values[y0 * size + x1] * wx;
    const float bottom = values[y1 * size + x0] * (1.0F - wx) +
        values[y1 * size + x1] * wx;
    return top * (1.0F - wy) + bottom * wy;
}

}  // namespace

std::vector<SegProposal> decode_proposals(
    const std::array<const float *, kSemanticOutputCount> &outputs,
    int input_size,
    float score_threshold
) {
    std::vector<SegProposal> proposals;
    float logit_threshold = -std::numeric_limits<float>::infinity();
    if (score_threshold >= 1.0F) {
        logit_threshold = std::numeric_limits<float>::infinity();
    } else if (score_threshold > 0.0F) {
        logit_threshold = std::log(score_threshold / (1.0F - score_threshold));
    }
    for (int scale = 0; scale < kScaleCount; ++scale) {
        const int stride = kStrides[scale];
        const int grid_width = input_size / stride;
        const int grid_size = grid_width * grid_width;
        const float *boxes = outputs[scale * 3];
        const float *scores = outputs[scale * 3 + 1];
        const float *coefficients = outputs[scale * 3 + 2];
        for (int y = 0; y < grid_width; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                const int index = y * grid_width + x;
                if (scores[index] < logit_threshold) {
                    continue;
                }
                const float score = sigmoid(scores[index]);
                if (score < score_threshold) {
                    continue;
                }
                SegProposal proposal{};
                proposal.x1 = (x + 0.5F - dfl(boxes, grid_size, index, 0)) * stride;
                proposal.y1 = (y + 0.5F - dfl(boxes, grid_size, index, 1)) * stride;
                proposal.x2 = (x + 0.5F + dfl(boxes, grid_size, index, 2)) * stride;
                proposal.y2 = (y + 0.5F + dfl(boxes, grid_size, index, 3)) * stride;
                proposal.x1 = std::clamp(proposal.x1, 0.0F, input_size - 1.0F);
                proposal.y1 = std::clamp(proposal.y1, 0.0F, input_size - 1.0F);
                proposal.x2 = std::clamp(proposal.x2, 0.0F, input_size - 1.0F);
                proposal.y2 = std::clamp(proposal.y2, 0.0F, input_size - 1.0F);
                proposal.score = score;
                for (int channel = 0; channel < kMaskChannels; ++channel) {
                    proposal.coefficients[channel] =
                        coefficients[channel * grid_size + index];
                }
                proposals.push_back(proposal);
            }
        }
    }
    return proposals;
}

std::vector<SegProposal> non_maximum_suppression(
    std::vector<SegProposal> proposals,
    float nms_threshold,
    int max_detections
) {
    std::sort(
        proposals.begin(), proposals.end(),
        [](const SegProposal &left, const SegProposal &right) {
            return left.score > right.score;
        }
    );
    std::vector<SegProposal> selected;
    for (const SegProposal &proposal : proposals) {
        const bool suppressed = std::any_of(
            selected.begin(), selected.end(),
            [&proposal, nms_threshold](const SegProposal &kept) {
                return iou(proposal, kept) > nms_threshold;
            }
        );
        if (!suppressed) {
            selected.push_back(proposal);
            if (static_cast<int>(selected.size()) >= max_detections) {
                break;
            }
        }
    }
    return selected;
}

void decode_mask_to_source(
    const SegProposal &proposal,
    const float *prototype,
    int prototype_size,
    const LetterboxGeometry &geometry,
    int source_width,
    int source_height,
    float mask_threshold,
    std::uint8_t *destination
) {
    const int prototype_area = prototype_size * prototype_size;
    std::vector<float> logits(prototype_area, 0.0F);
    for (int channel = 0; channel < kMaskChannels; ++channel) {
        const float weight = proposal.coefficients[channel];
        const float *plane = prototype + channel * prototype_area;
        for (int index = 0; index < prototype_area; ++index) {
            logits[index] += weight * plane[index];
        }
    }
    const float threshold_logit = mask_threshold <= 0.0F
        ? -std::numeric_limits<float>::infinity()
        : (mask_threshold >= 1.0F
            ? std::numeric_limits<float>::infinity()
            : std::log(mask_threshold / (1.0F - mask_threshold)));
    std::fill(
        destination,
        destination + static_cast<std::size_t>(source_width) * source_height,
        static_cast<std::uint8_t>(0)
    );
    const int minimum_x = std::clamp(
        static_cast<int>(std::floor((proposal.x1 - geometry.left) / geometry.scale)),
        0, source_width - 1
    );
    const int maximum_x = std::clamp(
        static_cast<int>(std::ceil((proposal.x2 - geometry.left) / geometry.scale)),
        0, source_width - 1
    );
    const int minimum_y = std::clamp(
        static_cast<int>(std::floor((proposal.y1 - geometry.top) / geometry.scale)),
        0, source_height - 1
    );
    const int maximum_y = std::clamp(
        static_cast<int>(std::ceil((proposal.y2 - geometry.top) / geometry.scale)),
        0, source_height - 1
    );
    const float model_to_prototype =
        static_cast<float>(prototype_size) / 640.0F;
    for (int y = minimum_y; y <= maximum_y; ++y) {
        const float model_y = static_cast<float>((y + 0.5) * geometry.scale + geometry.top);
        const float prototype_y = model_y * model_to_prototype - 0.5F;
        for (int x = minimum_x; x <= maximum_x; ++x) {
            const float model_x = static_cast<float>((x + 0.5) * geometry.scale + geometry.left);
            const float prototype_x = model_x * model_to_prototype - 0.5F;
            if (bilinear(logits, prototype_size, prototype_x, prototype_y) >= threshold_logit) {
                destination[static_cast<std::size_t>(y) * source_width + x] = 255;
            }
        }
    }
}

}  // namespace ec26e::vision2
