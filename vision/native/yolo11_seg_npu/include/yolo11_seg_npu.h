#ifndef EC26E_YOLO11_SEG_NPU_H
#define EC26E_YOLO11_SEG_NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YoloSegDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int32_t class_id;
} YoloSegDetection;

typedef struct YoloSegNativeTiming {
    uint64_t preprocess_ns;
    uint64_t input_flush_ns;
    uint64_t network_ns;
    uint64_t output_map_ns;
    uint64_t detection_decode_ns;
    uint64_t nms_ns;
    uint64_t mask_decode_ns;
    uint64_t output_unmap_ns;
    uint64_t total_ns;
} YoloSegNativeTiming;

void *yolo_seg_create(
    const char *model_path,
    float score_threshold,
    float nms_threshold,
    float mask_threshold
);

int yolo_seg_infer_bgr(
    void *context,
    const uint8_t *bgr_data,
    size_t bgr_size,
    int width,
    int height,
    size_t row_stride,
    YoloSegDetection *detections,
    int max_detections,
    uint8_t *masks,
    size_t masks_size
);

int yolo_seg_input_size(void *context);
int yolo_seg_last_timing(void *context, YoloSegNativeTiming *timing);
uint32_t yolo_seg_driver_version(void *context);
const char *yolo_seg_last_error(void *context);
void yolo_seg_destroy(void *context);

#ifdef __cplusplus
}
#endif

#endif
