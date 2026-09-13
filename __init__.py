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

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
