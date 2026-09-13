# -*- coding: utf-8 -*-
"""
BSAI Face Refine —— 一键人脸高清修复节点
=========================================

把官方 ComfyUI-H3-FaceRefine 的完整链路（人脸跟踪裁剪 -> H3 潜空间重绘 ->
逐帧去噪 -> 缝合回原视频）封装成**单个节点**，并自动处理多人场景：

- 不提供身份参考图：自动跟踪视频中最大的人脸修复一次；
- 提供 1~4 张身份参考图：依次为每个人跑一遍完整链路，并把上一遍的
  缝合结果作为下一遍的底图（官方 README 推荐的多人链式累积做法）。

内部流程（每个主体一遍）：
    H3FaceTrackCrop         检测 + 逐帧裁剪 + 身份跟踪
    MiniMaxH3ReferenceToVideo   构造 H3 conditioning（可选身份参考注入）
    H3InjectVideoLatent     把真实裁剪帧编码进 H3 AV latent（img2img 起点）
    MiniMaxH3NativeAudioLock   （可选）锁定原视频音频/口型
    H3PerFrameDenoise       按人脸大小逐帧调节 denoise 强度
    RandomNoise/BasicGuider/BasicScheduler/KSamplerSelect/SamplerCustomAdvanced
    VAEDecode               重绘后的裁剪序列解码
    [可选第二阶段] BSAI-H3-upscale-4K 的 VOSR / CodeFormer 细节增强
    H3FaceStitch            把重绘人脸羽化缝合回底图

第二阶段（对应参考视频 BV1oXYL6ZE9V 的 VOSR2 细节修复）复用本机
BSAI-H3-upscale-4K 插件引擎：VOSR 2.0 生成式细节修复 / CodeFormer /
GFPGAN。VOSR 最低 2 倍生成，缝合层按归一化 affine 网格贴回原坐标，
等效"只修不放大"且细节保留更多。
"""

import os
import sys
import importlib.util

# ---------------------------------------------------------------------------
# 兼容加载：本机已安装的官方 H3 插件模块（目录名含连字符，无法直接 import）
# 优先复用 ComfyUI 已加载的模块实例，保证与界面注册的是同一份节点类。
# ---------------------------------------------------------------------------

_CANDIDATE_SUBSTRINGS = {
    "h3fr": ("h3-facerefine", "h3facerefine", "facerefine"),
    "nal": ("nativelock", "native_audio", "h3-nativeaudio"),
    "bsai4k": ("bsai-h3-upscale-4k", "h3-upscale-4k", "upscale_4k", "bsai_h3_upscale"),
}


def _find_loaded_module(*substrings):
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        low = (getattr(mod, "__name__", "") or "").lower()
        if not low:
            continue
        if any(s in low for s in substrings):
            return mod
    return None


def _load_file_module(path, mod_name):
    """用独立模块名加载一个 .py 文件（不与 ComfyUI 已加载模块冲突）。"""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HFR_MAPPINGS = None
_HFR_MOD = None


def _get_hfr():
    """返回 H3FaceRefine 的 NODE_CLASS_MAPPINGS。"""
    global _HFR_MAPPINGS, _HFR_MOD
    if _HFR_MAPPINGS is not None:
        return _HFR_MAPPINGS

    mod = _find_loaded_module(*_CANDIDATE_SUBSTRINGS["h3fr"])
    if mod is None or not hasattr(mod, "NODE_CLASS_MAPPINGS"):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "ComfyUI-H3-FaceRefine", "nodes.py")
        if os.path.isfile(path):
            mod = _load_file_module(path, "bsai_h3fr_nodes")
    if mod is None or not hasattr(mod, "NODE_CLASS_MAPPINGS"):
        raise RuntimeError(
            "BSAI Face Refine: 找不到 ComfyUI-H3-FaceRefine 节点集。"
            "请确认该插件已安装在 custom_nodes 下。"
        )
    _HFR_MOD = mod
    _HFR_MAPPINGS = mod.NODE_CLASS_MAPPINGS
    return _HFR_MAPPINGS


_NAL_MOD = None


def _get_audio_lock():
    """返回 MiniMaxH3NativeAudioLock 类；不可用时返回 None。"""
    global _NAL_MOD
    if _NAL_MOD is not None:
        return _NAL_MOD
    mod = _find_loaded_module(*_CANDIDATE_SUBSTRINGS["nal"])
    if mod is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "ComfyUI-H3-NativeAudioLock", "__init__.py")
        if os.path.isfile(path):
            try:
                mod = _load_file_module(path, "bsai_h3_natlock")
            except Exception:
                mod = None
    if mod is not None and hasattr(mod, "MiniMaxH3NativeAudioLock"):
        _NAL_MOD = mod.MiniMaxH3NativeAudioLock
        return _NAL_MOD
    _NAL_MOD = None
    return None


# ---------------------------------------------------------------------------
# 采样/conditioning：ComfyUI 0.35 新版 schema 节点（comfy_extras 内置）
# ---------------------------------------------------------------------------

def _import_custom_sampler():
    try:
        from comfy_extras import nodes_custom_sampler as m
        return m
    except Exception as e:
        raise RuntimeError("BSAI Face Refine: 无法导入 comfy_extras.nodes_custom_sampler: %s" % e)


def _import_minimax_h3():
    try:
        from comfy_extras import nodes_minimax_h3 as m
        return m
    except Exception as e:
        raise RuntimeError("BSAI Face Refine: 无法导入 comfy_extras.nodes_minimax_h3: %s" % e)


# ---------------------------------------------------------------------------
# 默认提示词（用户可覆盖）。H3 原生以英文提示词效果最好。
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = (
    "ultra high definition close-up of a human face, crystal clear facial details, "
    "natural realistic skin texture, sharp detailed eyes, correct natural facial "
    "features, photorealistic, professional portrait quality"
)


def _align_h3_len(n):
    """H3 latent 帧数按 17k+5 网格对齐（与官方节点一致）。"""
    if n < 5:
        return max(5, n)
    k = max(0, (n - 5 + 16) // 17)
    return k * 17 + 5


# ---------------------------------------------------------------------------
# 第二阶段：复用本机 BSAI-H3-upscale-4K 引擎（VOSR 2.0 / CodeFormer / GFPGAN）
# ---------------------------------------------------------------------------

_STAGE2_ENGINES = [
    # ---- 快速档（推荐日常使用）----
    "1xSkinContrast (皮肤细节·极速)",
    "RealESRGAN_x2plus (通用2倍·快速)",
    "DLSS 5 (RTX硬件加速·快速)",
    "FlashVSR (扩散视频超分·快速)",
    "SeedVR2 7B INT8 (高质量·中速)",
    # ---- 极致档 ----
    "VOSR 2.0 (CVPR2026生成式超分)",
    # ---- 人脸重建档 ----
    "CodeFormer",
    "GFPGANv1.4",
    "小脸增强(CodeFormer)",
]

# 引擎显示名 -> BSAI-H3-upscale-4K 的 model_name（ENGINE_OPTIONS 键或模型文件名）
_STAGE2_ENGINE_MODEL = {
    "1xSkinContrast (皮肤细节·极速)": "1xSkinContrast-SuperUltraCompact.pth",
    "RealESRGAN_x2plus (通用2倍·快速)": "RealESRGAN_x2plus.pth",
    "DLSS 5 (RTX硬件加速·快速)": "DLSS 5 (NVIDIA 神经渲染超分)",
    "FlashVSR (扩散视频超分·快速)": "FlashVSR-v1.1 (扩散视频超分)",
    "SeedVR2 7B INT8 (高质量·中速)": "SeedVR2 7B INT8 (ComfyUI原生)",
    "VOSR 2.0 (CVPR2026生成式超分)": "VOSR 2.0 (CVPR2026生成式超分)",
}

_FACE_RESTORE_ENGINES = ("CodeFormer", "GFPGANv1.4", "小脸增强(CodeFormer)")


_BSAI4K_MOD = None


def _get_bsai_4k():
    """加载本机 BSAI-H3-upscale-4K 插件模块（VOSR/CodeFormer 引擎宿主）。"""
    global _BSAI4K_MOD
    if _BSAI4K_MOD is not None:
        return _BSAI4K_MOD
    mod = _find_loaded_module(*_CANDIDATE_SUBSTRINGS["bsai4k"])
    if mod is not None:
        _BSAI4K_MOD = mod
        return mod
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("BSAI-H3-upscale-4K", "BSAI-H3-upscale-4k", "BSAI-H3-Upscale-4K"):
        path = os.path.join(base, rel, "bsai_h3_upscale_4k.py")
        if os.path.isfile(path):
            try:
                _BSAI4K_MOD = _load_file_module(path, "bsai_h3_upscale_4k_shared")
                return _BSAI4K_MOD
            except Exception as e:
                raise RuntimeError("BSAI Face Refine: BSAI-H3-upscale-4K 加载失败: %s" % e)
    raise RuntimeError(
        "BSAI Face Refine: 未找到 BSAI-H3-upscale-4K 插件（第二阶段 VOSR/CodeFormer 引擎）。"
        "请确认 custom_nodes\\BSAI-H3-upscale-4K 已安装。"
    )


def _stage2_enhance(crops, engine, scale, cfg, steps, fidelity, blend, temporal):
    """对 H3 重绘后的裁剪序列做第二阶段细节增强。

    - VOSR / 快速超分引擎（FlashVSR / SeedVR2 INT8 / DLSS5 / RealESRGAN /
      1xSkinContrast）：调 BSAI_H3_Upscale4K 对应引擎。缝合层按 transform
      贴回原坐标 -> 等效"只修不放大"。
    - CodeFormer / GFPGAN：调 BSAI_H3_FaceRestore（轻量人脸重建）。
    返回 (enhanced_crops, info)。
    """
    if crops is None or crops.shape[0] == 0:
        return crops, "stage2: skip (empty crops)"
    mod = _get_bsai_4k()
    if engine == "VOSR 2.0 (CVPR2026生成式超分)":
        node = mod.BSAI_H3_Upscale4K()
        scale = max(1.0, float(scale))
        out, w, h, eff, info = node.upscale(**{
            "images / 图像": crops,
            "model_name / 模型": "VOSR 2.0 (CVPR2026生成式超分)",
            "scale / 放大倍数": scale,
            "vosr_cfg / VOSR保真度": float(cfg),
            "vosr_steps / VOSR步数": int(steps),
            "face_restore / 人脸修复": "Off",
            "detail_mode / 细节模式": "smart",
            "input_adaptive / 输入自适应": "关",
            "auto_prefer / 自动路由偏好": "质量优先",
        })
        return out, "stage2: VOSR 2.0 x%.2f -> %dx%d | %s" % (eff, w, h, info)
    if engine in _FACE_RESTORE_ENGINES:
        node = mod.BSAI_H3_FaceRestore()
        out, nf, info = node.restore(**{
            "images / 图像": crops,
            "face_restore / 人脸修复": engine,
            "face_det_conf / 检测置信度": 0.15,
            "face_blend / 融合强度": float(blend),
            "face_fidelity / 保真度": float(fidelity),
            "face_temporal / 人脸时域稳定": float(temporal),
        })
        return out, "stage2: %s (faces=%d) | %s" % (engine, nf, info)
    # ---- 快速超分档（FlashVSR / SeedVR2 INT8 / DLSS5 / RealESRGAN / 1x 模型）----
    model_name = _STAGE2_ENGINE_MODEL.get(engine)
    if model_name is None:
        raise RuntimeError("未知第二阶段引擎: %s" % engine)
    # 1x 模型 = 只修不放大；其余按用户 stage2_scale
    s = 1.0 if model_name.lower().startswith("1x") else max(1.0, float(scale))
    node = mod.BSAI_H3_Upscale4K()
    out, w, h, eff, info = node.upscale(**{
        "images / 图像": crops,
        "model_name / 模型": model_name,
        "scale / 放大倍数": s,
        "temporal_strength / 时序强度": 0.2,
        "detail_amount / 细节强度": 0.5,
        "detail_radius / 细节半径": 1.8,
        "softness / 柔和度": 0.1,
        "face_restore / 人脸修复": "Off",
        "input_adaptive / 输入自适应": "关",
        "auto_prefer / 自动路由偏好": "质量优先",
    })
    return out, "stage2: %s x%.2f -> %dx%d | %s" % (engine, eff, w, h, info)


def _detector_choices():
    try:
        return _get_hfr()["H3FaceTrackCrop"].INPUT_TYPES()["required"]["detector"][0]
    except Exception:
        return ["bbox\\face_yolov8m.pt", "bbox\\face_yolov8n.pt", "bbox\\face_yolov8s.pt", "bbox\\face_yolov9c.pt"]


def _find_person_detector():
    """定位本机可用的 person 全身分割模型，用作远景小脸的 fallback_detector。

    视频（Benji's AI Playground / 中译）的核心结论：远景镜头里人脸小到看不清时，
    死盯人脸框只会让 H3 脑补出结构崩坏的脸；改用 person 全身模型把整个人物作为
    整体框，人脸检测器丢帧时用人体框顶部反推头部位置，远胜盲目插值。
    优先 segm/person_yolov8m-seg.pt，其次 bbox 下同名/小模型。
    """
    candidates = [
        "person_yolov8m-seg.pt",
        "person_yolov8s-seg.pt",
        "person_yolov8n-seg.pt",
        "person_detect_v0_s_yv11.pt",
    ]
    try:
        import folder_paths as _fp
        for key in ("ultralytics_bbox", "ultralytics"):
            try:
                names = set(_fp.get_filename_list(key))
            except Exception:
                names = set()
            for c in candidates:
                if c in names:
                    return c
            for c in candidates:
                p = _fp.get_full_path(key, c)
                if p:
                    return c
    except Exception:
        pass
    # 退回到标准 models 树直接探测
    base = getattr(folder_paths, "models_dir", None)
    if base:
        for sub in ("ultralytics/segm", "ultralytics/bbox"):
            for c in candidates:
                cand = os.path.join(base, *sub.split("/"), c)
                if os.path.isfile(cand):
                    return c
    return None


class BSAIFaceRefine:
    """BSAI Face Refine —— 一键人脸高清修复（单人 / 多人自动）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "源视频帧序列（任意长度，建议完整帧率）"}),
                "model": ("MODEL", {"tooltip": "MiniMax H3 视频模型（建议使用 TURBO 4 步 / 8 步模型）"}),
                "clip": ("CLIP", {"tooltip": "H3 文本编码器（Qwen3VL-MiniMax-H3）"}),
                "vae": ("VAE", {"tooltip": "H3 视频 VAE"}),
                "audio_vae": ("VAE", {"tooltip": "H3 音频 VAE（启用音频锁定时必需）"}),
                "detector": (_detector_choices(), {"tooltip": "人脸检测模型（models/ultralytics/bbox 下）"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1, "tooltip": "H3 重绘采样步数（参考工作流 8 步）"}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "人脸重绘强度（作用于大脸帧）。小脸帧自动用全强度（参考工作流语义：大脸 0.35、小脸 1.0）。崩坏严重可到 0.5~0.55"}),
                "canvas_size": ("INT", {"default": 768, "min": 512, "max": 2048, "step": 64,
                                        "tooltip": "人脸裁剪画布尺寸（正方形）。768 为 H3 原生推荐；越大越清晰但越耗显存"}),
                "small_face_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "小脸帧的重绘强度（参考工作流 1.0）。帧间按人脸大小平滑过渡"}),
                "confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05}),
                "crop_factor": ("FLOAT", {"default": 2.5, "min": 1.2, "max": 6.0, "step": 0.1,
                                          "tooltip": "人脸框外扩倍数，给重绘提供上下文"}),
                "smooth_window": ("INT", {"default": 21, "min": 1, "max": 121, "step": 2,
                                          "tooltip": "裁剪位置时域平滑窗口"}),
                "scheduler": (["beta", "simple", "karras", "sgm_uniform", "exponential"], {"default": "beta"}),
                "sampler_name": (["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "uni_pc", "heun"], {"default": "euler"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 53 - 1, "step": 1}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPT, "multiline": True,
                                      "tooltip": "重绘提示词。默认是通用高清人像增强词，可自定义"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "原视频音频（启用后用于口型/音频锁定，保持音画同步）"}),
                "identity_ref_1": ("IMAGE", {"tooltip": "第 1 个人的身份参考图（单人修复可留空=自动跟踪最大脸）"}),
                "identity_ref_2": ("IMAGE", {"tooltip": "第 2 个人的身份参考图（多人场景）"}),
                "identity_ref_3": ("IMAGE", {"tooltip": "第 3 个人的身份参考图（多人场景）"}),
                "identity_ref_4": ("IMAGE", {"tooltip": "第 4 个人的身份参考图（多人场景）"}),
                "enable_audio_lock": ("BOOLEAN", {"default": True, "tooltip": "用原视频音频锁定口型（需 audio_vae + audio）"}),
                "identity_threshold": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "paste_region": (["face_only", "face_ellipse", "full_crop"], {"default": "face_only"}),
                "mask_dilation": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2}),
                "colour_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                # ---- 第二阶段（可选）：VOSR / CodeFormer 细节增强 ----
                "stage2_enable": ("BOOLEAN", {"default": False,
                                              "tooltip": "开启第二阶段细节增强（复用本机 BSAI-H3-upscale-4K 引擎，对应参考视频的 VOSR2 环节）"}),
                "stage2_engine": (_STAGE2_ENGINES, {"default": "1xSkinContrast (皮肤细节·极速)",
                                                    "tooltip": "第二阶段引擎：快速档(1xSkinContrast皮肤细节/RealESRGAN/DLSS5/FlashVSR/SeedVR2-INT8)、极致档(VOSR生成式细节)或人脸重建(CodeFormer/GFPGAN)"}),
                "stage2_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5,
                                           "tooltip": "第二阶段采样倍率。1x 引擎(1xSkinContrast/1xDeJPG)自动固定 1 倍只修不放大；其余引擎按此倍率生成，缝合层按原坐标贴回。VOSR 最低 2 倍"}),
                "stage2_cfg": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.1,
                                         "tooltip": "VOSR / SeedVR2 保真度（越高越忠于原脸）"}),
                "stage2_steps": ("INT", {"default": 1, "min": 1, "max": 25, "step": 1,
                                         "tooltip": "VOSR 推理步数（1 = 一步生成最快；纹理不满意可到 2~4）"}),
                "stage2_fidelity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "CodeFormer 保真度（仅 CodeFormer 引擎使用）"}),
                "stage2_blend": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                                           "tooltip": "CodeFormer 融合强度（仅 CodeFormer 引擎使用）"}),
                "stage2_temporal": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "人脸时域稳定（仅 CodeFormer 引擎使用）"}),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"], {"default": "fade_out"}),
                "person_fallback": ("BOOLEAN", {"default": False,
                    "tooltip": "远景小脸兜底：仅当人脸检测器在某些帧彻底丢帧时，用 person 全身分割模型"
                               "从人体框顶部反推头部。注意：它会参与轨迹平滑，中景/夜景镜头若估算偏低会把裁剪框拉偏到脖子导致噪点。"
                               "建议默认关；只在确认纯远景小脸、且远景帧脸框频繁丢失时手动开。需 models/ultralytics/segm/person_yolov8m-seg.pt。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "report", "face_count")
    FUNCTION = "run"
    CATEGORY = "BSAI/H3 Face Refine"
    DESCRIPTION = "一键修复视频人脸变形/崩坏/模糊，单人多人自动处理，输出高清人脸。"

    def run(
        self,
        images,
        model,
        clip,
        vae,
        audio_vae,
        detector,
        steps,
        denoise,
        canvas_size,
        small_face_denoise,
        confidence,
        crop_factor,
        smooth_window,
        scheduler,
        sampler_name,
        seed,
        prompt,
        audio=None,
        identity_ref_1=None,
        identity_ref_2=None,
        identity_ref_3=None,
        identity_ref_4=None,
        enable_audio_lock=True,
        identity_threshold=0.28,
        paste_region="face_only",
        mask_dilation=24,
        feather=24,
        colour_match=1.0,
        blend=1.0,
        stage2_enable=False,
        stage2_engine="1xSkinContrast (皮肤细节·极速)",
        stage2_scale=2.0,
        stage2_cfg=0.5,
        stage2_steps=1,
        stage2_fidelity=0.6,
        stage2_blend=0.7,
        stage2_temporal=0.5,
        undetected_frames="fade_out",
        person_fallback=False,
    ):
        hfr = _get_hfr()
        H3FaceTrackCrop = hfr["H3FaceTrackCrop"]
        H3FaceStitch = hfr["H3FaceStitch"]
        H3InjectVideoLatent = hfr["H3InjectVideoLatent"]
        H3PerFrameDenoise = hfr["H3PerFrameDenoise"]

        nmh = _import_minimax_h3()
        csm = _import_custom_sampler()
        MiniMaxH3ReferenceToVideo = nmh.MiniMaxH3ReferenceToVideo
        BasicScheduler = csm.BasicScheduler
        KSamplerSelect = csm.KSamplerSelect
        BasicGuider = csm.BasicGuider
        RandomNoise = csm.RandomNoise
        SamplerCustomAdvanced = csm.SamplerCustomAdvanced

        audio_lock_cls = _get_audio_lock()

        refs = [r for r in (identity_ref_1, identity_ref_2, identity_ref_3, identity_ref_4) if r is not None]
        if not refs:
            refs = [None]  # 无身份参考 -> 自动跟踪最大脸

        base = images
        reports = []
        face_count = 0

        # ---- 远景小脸兜底：person 全身模型 fallback_detector -------------------
        # 视频（Benji's AI Playground）核心结论：远景镜头人脸小到看不清时，人脸检测器
        # 频繁丢帧，盲目插值会导致跟踪框在脸外乱跳、H3 脑补出错位五官。启用后，丢帧时
        # 用 person 全身分割框顶部反推头部，跟踪连续且稳定。
        person_model = _find_person_detector() if person_fallback else None
        if person_fallback and person_model:
            print("[BSAIFaceRefine-v5] person fallback detector = %s（远景小脸丢帧时用人体框定位头部）" % person_model)
            fb_detector = person_model
        elif person_fallback:
            print("[BSAIFaceRefine-v5] 已开 person_fallback 但未找到 person 分割模型；"
                  "远景小脸丢帧将退化为插值。可在 models/ultralytics/segm 放 person_yolov8m-seg.pt")
            fb_detector = "none"
        else:
            fb_detector = "none"

        for pass_idx, ref in enumerate(refs):
            # 版本标记：确认 ComfyUI 进程加载的是最新代码（日志搜索 BSAIFaceRefine-v3）
            if pass_idx == 0:
                print("[BSAIFaceRefine-v5] NestedTensor 解码 + 远景小脸自动降倍已启用")
            _small_face_mode = False
            _eff_small_denoise = small_face_denoise
            # ---- 1. 检测 + 逐帧裁剪 + 身份跟踪 --------------------------------
            track = H3FaceTrackCrop()
            crops, transform, preview, track_report, cw, ch, K = track.run(
                images=base,
                detector=detector,
                confidence=confidence,
                crop_factor=crop_factor,
                canvas_width=canvas_size,
                canvas_height=canvas_size,
                canvas_mode="manual",
                smooth_window=smooth_window,
                size_smooth_window=51,
                smooth_method="gaussian",
                size_mode="per_frame",
                select="largest_face",
                identity_reference=ref,
                identity_threshold=identity_threshold,
                identity_track=True,
                identity_model="insightface",
                cut_detection="none",
                fallback_detector=fb_detector,
                fallback_head_frac=0.5,
            )
            if crops is None:
                reports.append("[pass %d] 未检测到人脸，跳过。" % (pass_idx + 1))
                continue

            face_count += 1
            reports.append(
                "[pass %d] 人脸 %d 个：裁剪画布 %dx%d，帧数 %d\n%s"
                % (pass_idx + 1, len(refs), cw, ch, K, (track_report or "")[:400])
            )

            # ---- 1.5 远景小脸自动降倍（防止 H3 放大过度导致结构崩坏） ----------
            # 源脸高 < 40px 且画布 > 512 时：9-13x 放大后 H3 无法从模糊裁剪合成完整脸，
            # 表现为"独眼/五官错位"。自动改用 512 画布 + crop_factor 3.5 降倍重裁，
            # 放大倍数降至 ~5-7x，给 H3 足够的像素合成完整五官。
            _face_h = [b[3] / max(crop_factor, 1.0) for b in transform["boxes"]] if transform.get("boxes") else []
            min_face = min(_face_h) if _face_h else None
            if min_face is not None and min_face < 40.0 and canvas_size > 512:
                _mag_old = canvas_size / (min_face * max(crop_factor, 1.0))
                reports.append(
                    "[小脸模式] 源脸高约 %.0fpx(<40px)，原放大 %.1fx 过大易致结构崩坏；"
                    "自动改 512 画布 + crop_factor 3.5 降倍重裁。" % (min_face, _mag_old))
                crops, transform, preview, track_report, cw, ch, K = track.run(
                    images=base, detector=detector, confidence=confidence,
                    crop_factor=3.5, canvas_width=512, canvas_height=512,
                    canvas_mode="manual", smooth_window=smooth_window,
                    size_smooth_window=51, smooth_method="gaussian", size_mode="per_frame",
                    select="largest_face", identity_reference=ref,
                    identity_threshold=identity_threshold, identity_track=True,
                    identity_model="insightface", cut_detection="none",
                    fallback_detector=fb_detector, fallback_head_frac=0.5,
                )
                canvas_size = 512
                _mag_new = 512.0 / (min_face * 3.5)
                reports.append("[小脸模式] 重裁完成：画布 512x512，放大 %.1fx，帧数 %d。" % (_mag_new, K))
                # 源脸仅 23~33px 时，放大 5~6 倍后几乎无五官信息。denoise 0.4 会让 H3
                # 自由重生成近半内容、在无参考图时发散成彩色噪点。压到 0.18：只做锐化/
                # 纹理增强，保留原糊脸结构，杜绝噪点。脸稍大（40~120px）走正常 0.35。
                _small_face_mode = True
                _eff_small_denoise = 0.18
                reports.append("[小脸模式] 小脸 denoise→0.18（23px 级糊脸只增强不重生成）：防 H3 发散成噪点。")

            # ---- 2. conditioning（可选身份参考注入） ---------------------------
            ref_images = None
            if ref is not None:
                ref_images = {"ref_image_1": ref}
            cond_len = _align_h3_len(int(K))
            res = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                prompt=prompt if prompt else _DEFAULT_PROMPT,
                width=canvas_size,
                height=canvas_size,
                length=cond_len,
                ref_image_size="match",
                vae=vae,
                audio_vae=audio_vae,
                ref_images=ref_images,
            )
            cond, av_latent = res.result

            # ---- 3. 注入真实裁剪帧（img2img 起点） ------------------------------
            latent, inject_report = H3InjectVideoLatent().run(av_latent, crops, vae)
            reports.append("  inject: %s" % (inject_report or "")[:300])

            # ---- 4. 音频锁定（口型同步） --------------------------------------
            active_model = model
            if audio is not None and enable_audio_lock and audio_lock_cls is not None:
                active_model, latent, _ = audio_lock_cls().lock_audio(model, latent, audio_vae, audio)
            elif audio is None and enable_audio_lock:
                reports.append("  [提示] 未连接 audio，音频锁定已跳过。")

            # ---- 5. 按人脸大小逐帧调节 denoise --------------------------------
            # 官方 H3PerFrameDenoise 返回 (av_latent, report, model)——RETURN_TYPES 顺序。
            patched_latent, pf_report, patched_model = H3PerFrameDenoise().run(
                active_model,
                latent,
                transform,
                denoise_multiplier_small_face=_eff_small_denoise,
                denoise_multiplier_large_face=denoise,
                face_px_small=30.0,
                face_px_large=120.0,
                gamma=1.0,
                smooth_frames=9,
                scale_mode="absolute_px",
            )
            reports.append("  per-frame denoise: %s" % (pf_report or "")[:200])

            # ---- 6. 采样（SamplerCustomAdvanced 链路） --------------------------
            sigmas = BasicScheduler.execute(patched_model, scheduler, steps, 1.0).result[0]
            sampler = KSamplerSelect.execute(sampler_name).result[0]
            guider = BasicGuider.execute(patched_model, cond).result[0]
            noise = RandomNoise.execute(seed + pass_idx * 1000003).result[0]
            refined_latent = SamplerCustomAdvanced.execute(
                noise, guider, sampler, sigmas, patched_latent
            ).result[0]

            # ---- 7. 解码重绘后的裁剪序列 --------------------------------------
            # H3 联合 AV latent 是 NestedTensor（视频+音频成员），先取视频成员再解码；
            # 解码输出 5D (B,T,H,W,C) 时折叠为 4D（与官方 VAEDecode 一致）。
            _samples = refined_latent["samples"]
            if getattr(_samples, "is_nested", False):
                _samples = _samples.unbind()[0]
            refined_crops = vae.decode(_samples)
            if len(refined_crops.shape) == 5:
                refined_crops = refined_crops.reshape(
                    -1, refined_crops.shape[-3], refined_crops.shape[-2], refined_crops.shape[-1])

            # ---- 7.5 第二阶段（可选）：VOSR / CodeFormer 细节增强 --------------
            if stage2_enable:
                try:
                    refined_crops, s2_info = _stage2_enhance(
                        refined_crops, stage2_engine,
                        scale=stage2_scale, cfg=stage2_cfg, steps=stage2_steps,
                        fidelity=stage2_fidelity, blend=stage2_blend,
                        temporal=stage2_temporal,
                    )
                    reports.append("[pass %d] 第二阶段: %s" % (pass_idx + 1, s2_info))
                except Exception as e:
                    reports.append("[pass %d] 第二阶段失败（已跳过，保留 H3 重绘结果）: %s"
                                   % (pass_idx + 1, e))

            # ---- 8. 羽化缝合回底图（多人时链式累积） ----------------------------
            (base,) = H3FaceStitch().run(
                base,
                refined_crops,
                transform,
                paste_region=paste_region,
                mask_dilation=mask_dilation,
                feather=feather,
                colour_match=colour_match,
                blend=blend,
                undetected_frames=undetected_frames,
            )
            reports.append("[pass %d] 缝合完成。" % (pass_idx + 1))

        full_report = "\n".join(reports)
        if not reports:
            full_report = "未检测到任何可修复的人脸。"
        return (base, full_report, face_count)


NODE_CLASS_MAPPINGS = {
    "BSAIFaceRefine": BSAIFaceRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIFaceRefine": "BSAI Face Refine (一键人脸高清修复)",
}
