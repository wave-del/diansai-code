"""Q6A HTP context, four-output YOLO11-seg decoding, no Torch dependency.

Import is inert. Only explicitly constructing an inferencer initializes the NPU.
The candidate is usable for local-image validation; motion acceptance is separate.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .contracts import NativeTiming, SegInstance, VisionConfig

MODEL_DIR = Path(__file__).resolve().parents[4] / "model-lib/q6a"
OUTPUT_SHAPES = ((1, 4, 8400), (1, 1, 8400), (1, 32, 8400), (1, 32, 160, 160))
_ENV_LOCK = threading.Lock()


def letterbox(image: np.ndarray) -> np.ndarray:
    """RGB UINT8 NCHW; fixed 640 input, centered 114 padding, same rounding as export."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("输入必须为 uint8 BGR numpy 图像")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("输入必须为非空 H×W×3 BGR 图像")
    h, w = image.shape[:2]
    gain = min(640 / h, 640 / w)
    rw, rh = round(w * gain), round(h * gain)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top, left = round((640-rh)/2-.1), round((640-rw)/2-.1)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = resized
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None])


def decode(outputs: tuple[np.ndarray, ...], shape: tuple[int, int], config: VisionConfig) -> tuple[SegInstance, ...]:
    """Decoded xywh and probabilities (no sigmoid/DFL); NMS before inverse mapping.

    All surviving proposals are returned. max_pieces is NOT a pre-dedup top-k.
    Score ties use anchor order to be deterministic on quantized outputs.
    """
    if len(outputs) != 4 or any(x.shape != s for x, s in zip(outputs, OUTPUT_SHAPES)):
        raise ValueError("QNN 四输出形状不符合 YOLO11n-seg 契约")
    if any(not np.isfinite(x).all() for x in outputs):
        raise ValueError("QNN 输出包含非有限数值")
    boxes, scores, coefficients, prototypes = outputs
    scores = scores.reshape(-1)
    candidates = np.flatnonzero(scores > config.score_threshold)
    if not len(candidates):
        return ()
    xywh = boxes[0].T
    xyxy = np.empty_like(xywh)
    xyxy[:, :2] = xywh[:, :2] - xywh[:, 2:]/2
    xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]/2
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    selected = []
    while len(order) and len(selected) < 300:
        current = int(order[0]); selected.append(current)
        rest = order[1:]
        intersection = np.maximum(np.minimum(xyxy[current, 2:], xyxy[rest, 2:]) -
                                  np.maximum(xyxy[current, :2], xyxy[rest, :2]), 0).prod(axis=1)
        a = np.maximum(xyxy[current, 2:]-xyxy[current, :2], 0).prod()
        b = np.maximum(xyxy[rest, 2:]-xyxy[rest, :2], 0).prod(axis=1)
        overlaps = intersection / np.maximum(a+b-intersection, 1e-12)
        order = rest[overlaps <= config.nms_threshold]
    h, w = shape
    gain = min(640/h, 640/w)
    pad_x = round((640-round(w*gain))/2-.1)
    pad_y = round((640-round(h*gain))/2-.1)
    selected_boxes = xyxy[selected].copy()
    selected_boxes -= np.asarray((pad_x,pad_y,pad_x,pad_y), dtype=np.float32)
    selected_boxes /= gain
    selected_boxes[:, (0,2)] = selected_boxes[:, (0,2)].clip(0,w)
    selected_boxes[:, (1,3)] = selected_boxes[:, (1,3)].clip(0,h)
    # Upsample logits, threshold, then crop: same order as process_mask_native.
    logits = coefficients[0][:, selected].T @ prototypes[0].reshape(32,-1)
    mgain = min(160/h,160/w)
    pw, ph = (160-round(w*mgain))/2, (160-round(h*mgain))/2
    top,left,bottom,right = round(ph-.1),round(pw-.1),160-round(ph+.1),160-round(pw+.1)
    threshold = (-math.inf if config.mask_threshold == 0 else math.inf if config.mask_threshold == 1
                 else math.log(config.mask_threshold/(1-config.mask_threshold)))
    result = []
    for anchor, box, low in zip(selected, selected_boxes, logits.reshape(-1,160,160)):
        full = cv2.resize(low[top:bottom,left:right], (w,h), interpolation=cv2.INTER_LINEAR)
        mask = (full > threshold).astype(np.uint8)
        x1,y1,x2,y2 = np.ceil(box).astype(int)
        mask[:y1] = 0; mask[y2:] = 0; mask[:, :x1] = 0; mask[:, x2:] = 0
        if np.any(mask):
            result.append(SegInstance(tuple(map(float,box)),float(scores[anchor]),0,mask))
    return tuple(result)


class QnnSession:
    """Own one native HTP graph and reusable output buffers on its creating thread."""
    def __init__(self, directory: Path = MODEL_DIR):
        self.directory = Path(directory).resolve()
        self._owner = threading.get_ident()
        self._handle = None
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest.get("backend") != "QNN HTP" or manifest.get("cpu_fallback") is not False:
            raise RuntimeError("仅允许已记录的 QNN HTP 资源包")
        required = {"libq6a_qnn.so", "puzzle_q6a.bin", "libQnnHtp.so", "libQnnSystem.so",
                    "libQnnHtpV68Stub.so", "libQnnHtpV68Skel.so"}
        if not required.issubset(manifest.get("sha256", {})):
            raise RuntimeError("Q6A 资源清单缺少必需文件校验项")
        for name, digest in manifest["sha256"].items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Q6A 部署文件校验失败：" + name)
        self.manifest = manifest
        lib = ctypes.CDLL(str(self.directory / "libq6a_qnn.so"))
        pointer = ctypes.c_void_p
        lib.q6a_create.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t]
        lib.q6a_create.restype = pointer
        lib.q6a_execute.argtypes = [pointer,pointer,ctypes.c_size_t,pointer,pointer,pointer,pointer,ctypes.POINTER(ctypes.c_double)]
        lib.q6a_execute.restype = ctypes.c_int
        lib.q6a_error.argtypes = [pointer]; lib.q6a_error.restype = ctypes.c_char_p
        lib.q6a_destroy.argtypes = [pointer]; lib.q6a_destroy.restype = None
        self._library = lib
        self._outputs = tuple(np.empty(shape, np.float32) for shape in OUTPUT_SHAPES)
        error = ctypes.create_string_buffer(512)
        # DSP search path is needed during context creation only. No shell/global changes.
        with _ENV_LOCK:
            previous = os.environ.get("ADSP_LIBRARY_PATH")
            os.environ["ADSP_LIBRARY_PATH"] = str(self.directory)
            try:
                self._handle = lib.q6a_create(os.fsencode(self.directory / "puzzle_q6a.bin"),
                                             os.fsencode(self.directory), error, len(error))
            finally:
                if previous is None: os.environ.pop("ADSP_LIBRARY_PATH", None)
                else: os.environ["ADSP_LIBRARY_PATH"] = previous
        if not self._handle:
            raise RuntimeError(error.value.decode(errors="replace"))
        self.executions = 0

    def _require_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("QNN 推理和释放必须在创建线程执行")
        if not self._handle:
            raise RuntimeError("QNN 会话已经关闭")

    def execute(self, nchw: np.ndarray) -> tuple[tuple[np.ndarray, ...], float]:
        self._require_owner()
        if nchw.dtype != np.uint8 or nchw.shape != (1,3,640,640):
            raise ValueError("QNN 输入必须为 UINT8 [1,3,640,640]")
        nchw = np.ascontiguousarray(nchw)
        milliseconds = ctypes.c_double()
        status = self._library.q6a_execute(self._handle,nchw.ctypes.data,nchw.size,
                         *[x.ctypes.data for x in self._outputs],ctypes.byref(milliseconds))
        if status:
            raise RuntimeError(self._library.q6a_error(self._handle).decode(errors="replace"))
        self.executions += 1
        # Copy so callers never retain buffers overwritten by the next frame.
        return tuple(x.copy() for x in self._outputs), milliseconds.value

    def close(self):
        if self._handle:
            self._require_owner()
            self._library.q6a_destroy(self._handle)
            self._handle = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class QnnYolo11SegInferencer:
    """Keep the project's infer/close/last_timing contract; no peripheral access."""
    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        model = Path(self.config.model_path)
        library = Path(self.config.library_path)
        if model.name != "puzzle_q6a.bin" or library.name != "libq6a_qnn.so":
            raise RuntimeError("Q6A 禁止加载 A733 .nb/VIPLite；请使用 model-lib/q6a 资源")
        directory = model.parent if model.is_absolute() else Path(__file__).resolve().parents[4] / model.parent
        library = library if library.is_absolute() else Path(__file__).resolve().parents[4] / library
        if library.resolve().parent != directory.resolve():
            raise ValueError("QNN Context 和桥接库必须来自同一已校验资源目录")
        self.session = QnnSession(directory)
        self.last_timing = NativeTiming()
        self.driver_version = (2 << 24) | (37 << 16) | (1 << 8)

    @property
    def input_size(self): return 640

    def infer(self, image_bgr: np.ndarray) -> tuple[SegInstance, ...]:
        self.session._require_owner()
        start = time.perf_counter()
        data = letterbox(image_bgr)
        pre = time.perf_counter()
        outputs, network = self.session.execute(data)
        executed = time.perf_counter()
        result = decode(outputs, image_bgr.shape[:2], self.config)
        end = time.perf_counter()
        self.last_timing = NativeTiming(preprocess_ms=(pre-start)*1000,network_ms=network,
             output_map_ms=max(0,(executed-pre)*1000-network),mask_decode_ms=(end-executed)*1000,total_ms=(end-start)*1000)
        return result

    def close(self): self.session.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
