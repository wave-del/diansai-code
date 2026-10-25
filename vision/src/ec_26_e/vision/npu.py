"""Existing vision interface backed by Q6A QNN HTP.

Historical VIPLite C++ sources remain in native/yolo11_seg_npu.
Importing this module never loads a native library or initializes the NPU.
"""
from .qnn import QnnYolo11SegInferencer as NpuYolo11SegInferencer

__all__ = ["NpuYolo11SegInferencer"]
