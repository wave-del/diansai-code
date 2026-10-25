#include "postprocess.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    std::array<std::vector<float>, ec26e::vision2::kSemanticOutputCount> storage;
    const std::array<int, 3> grids{80, 40, 20};
    for (int scale = 0; scale < 3; ++scale) {
        const int area = grids[scale] * grids[scale];
        storage[scale * 3].assign(64 * area, -20.0F);
        storage[scale * 3 + 1].assign(area, -20.0F);
        storage[scale * 3 + 2].assign(32 * area, 0.0F);
    }
    storage[9].assign(32 * 160 * 160, 0.0F);
    const int index = 10 * 80 + 10;
    storage[1][index] = 8.0F;
    for (int side = 0; side < 4; ++side) {
        storage[0][(side * 16 + 2) * 6400 + index] = 20.0F;
    }
    storage[2][index] = 1.0F;
    storage[9].assign(32 * 160 * 160, 1.0F);

    std::array<const float *, ec26e::vision2::kSemanticOutputCount> outputs{};
    for (std::size_t i = 0; i < outputs.size(); ++i) outputs[i] = storage[i].data();
    const auto proposals = ec26e::vision2::decode_proposals(outputs, 640, 0.25F);
    assert(proposals.size() == 1);
    assert(std::abs(proposals[0].x1 - 68.0F) < 0.1F);
    assert(std::abs(proposals[0].x2 - 100.0F) < 0.1F);
    auto duplicate = proposals;
    duplicate.push_back(proposals[0]);
    assert(ec26e::vision2::non_maximum_suppression(duplicate, 0.45F, 4).size() == 1);

    std::vector<std::uint8_t> mask(640 * 640, 0);
    const ec26e::vision2::LetterboxGeometry geometry{1.0, 640, 640, 0, 0};
    ec26e::vision2::decode_mask_to_source(
        proposals[0], storage[9].data(), 160, geometry,
        640, 640, 0.5F, mask.data()
    );
    assert(mask[80 * 640 + 80] == 255);
    assert(mask[10 * 640 + 10] == 0);
    return 0;
}
