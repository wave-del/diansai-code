#include "yolo11_seg_npu.h"

#include "postprocess.h"
#include "preprocess.h"

#include <vip_lite.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr int kInputSize = 640;
constexpr int kInputChannels = 3;
constexpr int kPrototypeSize = 160;

thread_local std::string global_last_error;
std::mutex runtime_mutex;
bool runtime_reserved = false;

struct Tensor {
    vip_buffer_create_params_t parameters{};
    vip_buffer buffer = VIP_NULL;
    vip_uint32_t elements = 0;
};

struct YoloSegContext {
    vip_network network = VIP_NULL;
    Tensor input;
    std::array<Tensor, ec26e::vision2::kSemanticOutputCount> outputs{};
    std::array<int, ec26e::vision2::kSemanticOutputCount> semantic_to_actual{};
    std::size_t input_byte_count = 0;
    float score_threshold = 0.25F;
    float nms_threshold = 0.45F;
    float mask_threshold = 0.5F;
    std::string last_error;
    std::thread::id owner_thread;
    bool runtime_initialized = false;
    bool network_prepared = false;
    bool owns_runtime_reservation = false;
    YoloSegNativeTiming last_timing{};

    ~YoloSegContext() { cleanup(); }

    bool fail(const std::string &message) {
        last_error = message;
        global_last_error = message;
        return false;
    }

    bool check(vip_status_e status, const char *operation) {
        if (status == VIP_SUCCESS) {
            return true;
        }
        std::ostringstream stream;
        stream << operation << " failed with VIPLite status "
               << static_cast<int>(status);
        return fail(stream.str());
    }

    bool require_owner_thread() {
        return std::this_thread::get_id() == owner_thread ||
            fail("VIPLite context must be used and destroyed on its creating thread");
    }

    void cleanup() {
        if (owner_thread != std::thread::id{} &&
            std::this_thread::get_id() != owner_thread) {
            global_last_error =
                "VIPLite context destruction attempted from a different thread";
            return;
        }
        if (network != VIP_NULL && network_prepared) {
            vip_finish_network(network);
            network_prepared = false;
        }
        if (input.buffer != VIP_NULL) {
            vip_destroy_buffer(input.buffer);
            input.buffer = VIP_NULL;
        }
        for (Tensor &output : outputs) {
            if (output.buffer != VIP_NULL) {
                vip_destroy_buffer(output.buffer);
                output.buffer = VIP_NULL;
            }
        }
        if (network != VIP_NULL) {
            vip_destroy_network(network);
            network = VIP_NULL;
        }
        if (runtime_initialized) {
            vip_destroy();
            runtime_initialized = false;
        }
        if (owns_runtime_reservation) {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            runtime_reserved = false;
            owns_runtime_reservation = false;
        }
    }
};

std::uint64_t elapsed_ns(
    const SteadyClock::time_point &started,
    const SteadyClock::time_point &finished = SteadyClock::now()
) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
            .count()
    );
}

vip_status_e query_tensor(
    vip_network network,
    bool input,
    vip_uint32_t index,
    Tensor &tensor
) {
    auto query = [network, input, index](vip_enum property, void *value) {
        return input ? vip_query_input(network, index, property, value)
                     : vip_query_output(network, index, property, value);
    };
    tensor.parameters.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_status_e status =
        query(VIP_BUFFER_PROP_DATA_FORMAT, &tensor.parameters.data_format);
    if (status != VIP_SUCCESS) return status;
    status = query(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &tensor.parameters.num_of_dims);
    if (status != VIP_SUCCESS) return status;
    status = query(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, tensor.parameters.sizes);
    if (status != VIP_SUCCESS) return status;
    status = query(VIP_BUFFER_PROP_QUANT_FORMAT, &tensor.parameters.quant_format);
    if (status != VIP_SUCCESS) return status;
    if (tensor.parameters.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        status = query(
            VIP_BUFFER_PROP_FIXED_POINT_POS,
            &tensor.parameters.quant_data.dfp.fixed_point_pos
        );
    } else if (tensor.parameters.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        status = query(
            VIP_BUFFER_PROP_TF_SCALE,
            &tensor.parameters.quant_data.affine.scale
        );
        if (status == VIP_SUCCESS) {
            status = query(
                VIP_BUFFER_PROP_TF_ZERO_POINT,
                &tensor.parameters.quant_data.affine.zeroPoint
            );
        }
    }
    if (status != VIP_SUCCESS) return status;
    tensor.elements = 1;
    for (vip_uint32_t dimension = 0;
         dimension < tensor.parameters.num_of_dims;
         ++dimension) {
        tensor.elements *= tensor.parameters.sizes[dimension];
    }
    return VIP_SUCCESS;
}

std::string shape_string(const Tensor &tensor) {
    std::ostringstream stream;
    stream << '[';
    for (vip_uint32_t dimension = 0;
         dimension < tensor.parameters.num_of_dims;
         ++dimension) {
        if (dimension != 0) stream << ',';
        stream << tensor.parameters.sizes[dimension];
    }
    stream << ']';
    return stream.str();
}

bool valid_threshold(float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

bool initialize_context(YoloSegContext &context, const char *model_path) {
    {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        if (runtime_reserved) {
            return context.fail(
                "only one A733 VIPLite YOLO segmentation context is supported per process"
            );
        }
        runtime_reserved = true;
        context.owns_runtime_reservation = true;
    }
    if (!context.check(vip_init(), "vip_init")) return false;
    context.runtime_initialized = true;
    if (!context.check(
            vip_create_network(
                model_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &context.network
            ),
            "vip_create_network"
        )) return false;

    vip_uint32_t input_count = 0;
    vip_uint32_t output_count = 0;
    if (!context.check(
            vip_query_network(
                context.network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count
            ),
            "query input count"
        ) ||
        !context.check(
            vip_query_network(
                context.network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count
            ),
            "query output count"
        )) return false;
    if (input_count != 1 ||
        output_count != ec26e::vision2::kSemanticOutputCount) {
        std::ostringstream stream;
        stream << "expected 1 input and 10 outputs, got " << input_count
               << " input(s) and " << output_count << " output(s)";
        return context.fail(stream.str());
    }

    if (!context.check(
            query_tensor(context.network, true, 0, context.input),
            "query input tensor"
        )) return false;
    const bool valid_input = context.input.parameters.num_of_dims == 4 &&
        context.input.parameters.sizes[0] == kInputChannels &&
        context.input.parameters.sizes[1] == kInputSize &&
        context.input.parameters.sizes[2] == kInputSize &&
        context.input.parameters.sizes[3] == 1 &&
        context.input.parameters.data_format == VIP_BUFFER_FORMAT_UINT8;
    if (!valid_input) {
        std::ostringstream stream;
        stream << "expected UINT8 input [3,640,640,1], got "
               << shape_string(context.input) << " format="
               << context.input.parameters.data_format;
        return context.fail(stream.str());
    }
    context.input_byte_count =
        static_cast<std::size_t>(kInputChannels) * kInputSize * kInputSize;
    if (context.input.elements != context.input_byte_count) {
        return context.fail("input tensor element count does not match its shape");
    }
    if (!context.check(
            vip_create_buffer(
                &context.input.parameters,
                sizeof(context.input.parameters),
                &context.input.buffer
            ),
            "create input buffer"
        )) return false;

    context.semantic_to_actual.fill(-1);
    for (vip_uint32_t index = 0; index < output_count; ++index) {
        Tensor &tensor = context.outputs[index];
        if (!context.check(
                query_tensor(context.network, false, index, tensor),
                "query output tensor"
            )) return false;
        if (tensor.parameters.data_format != VIP_BUFFER_FORMAT_FP32 ||
            tensor.parameters.num_of_dims != 4 ||
            tensor.parameters.sizes[3] != 1) {
            std::ostringstream stream;
            stream << "unexpected output " << index << " shape "
                   << shape_string(tensor) << " format="
                   << tensor.parameters.data_format;
            return context.fail(stream.str());
        }
        const int width = static_cast<int>(tensor.parameters.sizes[0]);
        const int height = static_cast<int>(tensor.parameters.sizes[1]);
        const int channels = static_cast<int>(tensor.parameters.sizes[2]);
        int semantic = -1;
        if (width == kPrototypeSize && height == kPrototypeSize &&
            channels == ec26e::vision2::kMaskChannels) {
            semantic = 9;
        } else {
            const std::array<int, 3> grids{80, 40, 20};
            for (int scale = 0; scale < 3; ++scale) {
                if (width != grids[scale] || height != grids[scale]) continue;
                int head = -1;
                if (channels == 64) head = 0;
                if (channels == 1) head = 1;
                if (channels == ec26e::vision2::kMaskChannels) head = 2;
                if (head >= 0) semantic = scale * 3 + head;
            }
        }
        if (semantic < 0 || context.semantic_to_actual[semantic] >= 0) {
            std::ostringstream stream;
            stream << "unexpected or duplicate output " << index << " shape "
                   << shape_string(tensor);
            return context.fail(stream.str());
        }
        context.semantic_to_actual[semantic] = static_cast<int>(index);
        if (!context.check(
                vip_create_buffer(
                    &tensor.parameters, sizeof(tensor.parameters), &tensor.buffer
                ),
                "create output buffer"
            )) return false;
    }
    if (std::any_of(
            context.semantic_to_actual.begin(),
            context.semantic_to_actual.end(),
            [](int value) { return value < 0; }
        )) return context.fail("model is missing one or more YOLO11-seg heads");

    if (!context.check(
            vip_prepare_network(context.network), "vip_prepare_network"
        )) return false;
    context.network_prepared = true;
    if (!context.check(
            vip_set_input(context.network, 0, context.input.buffer),
            "vip_set_input"
        )) return false;
    for (vip_uint32_t index = 0; index < output_count; ++index) {
        if (!context.check(
                vip_set_output(context.network, index, context.outputs[index].buffer),
                "vip_set_output"
            )) return false;
    }
    return true;
}

float restore_coordinate(float coordinate, int padding, double scale, int extent) {
    return std::clamp(
        static_cast<float>((coordinate - padding) / scale),
        0.0F,
        static_cast<float>(extent - 1)
    );
}

int run_inference(
    YoloSegContext &context,
    YoloSegDetection *detections,
    int max_detections,
    std::uint8_t *masks,
    const ec26e::vision2::LetterboxGeometry &geometry,
    int source_width,
    int source_height,
    const SteadyClock::time_point &total_started
) {
    auto started = SteadyClock::now();
    if (!context.check(
            vip_flush_buffer(context.input.buffer, VIP_BUFFER_OPER_TYPE_FLUSH),
            "flush input buffer"
        )) return -1;
    context.last_timing.input_flush_ns = elapsed_ns(started);

    started = SteadyClock::now();
    if (!context.check(vip_run_network(context.network), "vip_run_network")) {
        return -1;
    }
    context.last_timing.network_ns = elapsed_ns(started);

    std::array<const float *, ec26e::vision2::kSemanticOutputCount> semantic{};
    std::array<void *, ec26e::vision2::kSemanticOutputCount> mapped{};
    started = SteadyClock::now();
    std::size_t mapped_count = 0;
    for (std::size_t slot = 0; slot < semantic.size(); ++slot) {
        Tensor &tensor = context.outputs[context.semantic_to_actual[slot]];
        if (!context.check(
                vip_flush_buffer(tensor.buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE),
                "invalidate output buffer"
            )) break;
        mapped[slot] = vip_map_buffer(tensor.buffer);
        if (mapped[slot] == nullptr) {
            context.fail("vip_map_buffer returned null for output");
            break;
        }
        semantic[slot] = static_cast<const float *>(mapped[slot]);
        ++mapped_count;
    }
    context.last_timing.output_map_ns = elapsed_ns(started);
    if (mapped_count != semantic.size()) {
        for (std::size_t slot = 0; slot < mapped_count; ++slot) {
            vip_unmap_buffer(
                context.outputs[context.semantic_to_actual[slot]].buffer
            );
        }
        return -1;
    }

    started = SteadyClock::now();
    std::vector<ec26e::vision2::SegProposal> proposals =
        ec26e::vision2::decode_proposals(
            semantic, kInputSize, context.score_threshold
        );
    context.last_timing.detection_decode_ns = elapsed_ns(started);

    started = SteadyClock::now();
    std::vector<ec26e::vision2::SegProposal> selected =
        ec26e::vision2::non_maximum_suppression(
            std::move(proposals), context.nms_threshold, max_detections
        );
    context.last_timing.nms_ns = elapsed_ns(started);

    started = SteadyClock::now();
    const std::size_t mask_plane =
        static_cast<std::size_t>(source_width) * source_height;
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const ec26e::vision2::SegProposal &source = selected[index];
        YoloSegDetection &destination = detections[index];
        destination.x1 = restore_coordinate(
            source.x1, geometry.left, geometry.scale, source_width
        );
        destination.y1 = restore_coordinate(
            source.y1, geometry.top, geometry.scale, source_height
        );
        destination.x2 = restore_coordinate(
            source.x2, geometry.left, geometry.scale, source_width
        );
        destination.y2 = restore_coordinate(
            source.y2, geometry.top, geometry.scale, source_height
        );
        destination.score = source.score;
        destination.class_id = 0;
        ec26e::vision2::decode_mask_to_source(
            source,
            semantic[9],
            kPrototypeSize,
            geometry,
            source_width,
            source_height,
            context.mask_threshold,
            masks + index * mask_plane
        );
    }
    context.last_timing.mask_decode_ns = elapsed_ns(started);

    started = SteadyClock::now();
    for (std::size_t slot = 0; slot < semantic.size(); ++slot) {
        vip_unmap_buffer(
            context.outputs[context.semantic_to_actual[slot]].buffer
        );
    }
    context.last_timing.output_unmap_ns = elapsed_ns(started);
    context.last_timing.total_ns = elapsed_ns(total_started);
    context.last_error.clear();
    return static_cast<int>(selected.size());
}

}  // namespace

extern "C" void *yolo_seg_create(
    const char *model_path,
    float score_threshold,
    float nms_threshold,
    float mask_threshold
) {
    global_last_error.clear();
    if (model_path == nullptr || *model_path == '\0') {
        global_last_error = "model path is empty";
        return nullptr;
    }
    if (!std::filesystem::is_regular_file(model_path)) {
        global_last_error = std::string("model file does not exist: ") + model_path;
        return nullptr;
    }
    if (!valid_threshold(score_threshold) || !valid_threshold(nms_threshold) ||
        !valid_threshold(mask_threshold)) {
        global_last_error = "score, NMS and mask thresholds must be between 0 and 1";
        return nullptr;
    }
    auto context = std::make_unique<YoloSegContext>();
    context->owner_thread = std::this_thread::get_id();
    context->score_threshold = score_threshold;
    context->nms_threshold = nms_threshold;
    context->mask_threshold = mask_threshold;
    if (!initialize_context(*context, model_path)) {
        global_last_error = context->last_error;
        return nullptr;
    }
    return context.release();
}

extern "C" int yolo_seg_infer_bgr(
    void *opaque_context,
    const std::uint8_t *bgr_data,
    std::size_t bgr_size,
    int width,
    int height,
    std::size_t row_stride,
    YoloSegDetection *detections,
    int max_detections,
    std::uint8_t *masks,
    std::size_t masks_size
) {
    const auto total_started = SteadyClock::now();
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    if (context == nullptr) {
        global_last_error = "YOLO segmentation context is null";
        return -1;
    }
    if (!context->require_owner_thread()) return -1;
    context->last_timing = {};
    if (bgr_data == nullptr || width <= 0 || height <= 0) {
        context->fail("BGR input pointer is null or dimensions are not positive");
        return -1;
    }
    if (detections == nullptr || max_detections <= 0 || masks == nullptr) {
        context->fail("detection or mask output buffer is invalid");
        return -1;
    }
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    if (plane > std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(max_detections)) {
        context->fail("requested mask output size overflows size_t");
        return -1;
    }
    const std::size_t required_masks = static_cast<std::size_t>(max_detections) * plane;
    if (
        masks_size < required_masks) {
        context->fail("detection or mask output buffer is invalid or too small");
        return -1;
    }

    auto started = SteadyClock::now();
    void *input_data = vip_map_buffer(context->input.buffer);
    if (input_data == nullptr) {
        context->fail("vip_map_buffer returned null for input");
        return -1;
    }
    ec26e::vision2::LetterboxGeometry geometry{};
    const bool prepared = ec26e::vision2::letterbox_bgr_to_rgb(
        bgr_data,
        bgr_size,
        width,
        height,
        row_stride,
        static_cast<std::uint8_t *>(input_data),
        context->input_byte_count,
        kInputSize,
        kInputSize,
        114,
        &geometry
    );
    vip_unmap_buffer(context->input.buffer);
    context->last_timing.preprocess_ns = elapsed_ns(started);
    if (!prepared) {
        context->fail("invalid BGR input buffer, dimensions, or row stride");
        return -1;
    }
    return run_inference(
        *context,
        detections,
        max_detections,
        masks,
        geometry,
        width,
        height,
        total_started
    );
}

extern "C" int yolo_seg_input_size(void *opaque_context) {
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    if (context == nullptr || !context->require_owner_thread()) return 0;
    return kInputSize;
}

extern "C" int yolo_seg_last_timing(
    void *opaque_context,
    YoloSegNativeTiming *timing
) {
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    if (context == nullptr || timing == nullptr ||
        !context->require_owner_thread()) return -1;
    *timing = context->last_timing;
    return 0;
}

extern "C" std::uint32_t yolo_seg_driver_version(void *opaque_context) {
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    if (context == nullptr || !context->require_owner_thread()) return 0;
    return vip_get_version();
}

extern "C" const char *yolo_seg_last_error(void *opaque_context) {
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    return context == nullptr ? global_last_error.c_str()
                              : context->last_error.c_str();
}

extern "C" void yolo_seg_destroy(void *opaque_context) {
    auto *context = static_cast<YoloSegContext *>(opaque_context);
    if (context == nullptr) return;
    if (!context->require_owner_thread()) {
        global_last_error = context->last_error;
        return;
    }
    delete context;
}
