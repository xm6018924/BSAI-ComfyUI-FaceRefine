# -*- coding: utf-8 -*-
"""
BSAI_ComfyUI_face_refine
========================
一键修复视频中人脸变形 / 崩坏 / 模糊，输出高清人脸。
单人、多人场景均支持：检测 -> 逐帧裁剪 -> H3 潜空间重绘 -> 逐帧去噪 -> 缝合回原视频。

依赖（本机已全部就绪）：
- ComfyUI-H3-FaceRefine（官方节点集，人脸检测/裁剪/缝合）
- ComfyUI-H3-NativeAudioLock（唇形/音频锁定，可选）
- MiniMax H3 视频模型 + VAE + CLIP（Qwen3VL-MiniMax-H3）
- ultralytics / insightface / onnxruntime（人脸检测与身份识别）
"""

from . import nodes as _nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 工作流直接引用官方 H3 FaceRefine 的类名（H3FaceTrackCrop）。本插件内嵌了该
# 模块但只注册了 BSAIFaceRefine，这里补注册这一个节点，旧工作流无需另装官方插件。
_TRACK_CROP = "H3FaceTrackCrop"
_native = _nodes._get_hfr()
if _TRACK_CROP in _native:
    NODE_CLASS_MAPPINGS.setdefault(_TRACK_CROP, _native[_TRACK_CROP])
    NODE_DISPLAY_NAME_MAPPINGS.setdefault(
        _TRACK_CROP, _nodes._HFR_MOD.NODE_DISPLAY_NAME_MAPPINGS[_TRACK_CROP])

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
