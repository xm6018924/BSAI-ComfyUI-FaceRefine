# BSAI-ComfyUI-FaceRefine

**一键修复 MiniMax H3 视频中变形 / 崩坏 / 模糊的人脸，单人多人通吃。**
**One-click refinement of deformed / broken / blurry faces in MiniMax H3 video — single or multi-person, fully automatic.**

把官方 `ComfyUI-H3-FaceRefine` 的整条链路（逐帧检测 → 人脸裁剪放大 → H3 潜空间低强度重绘 → 逐帧去噪 → 羽化缝合回原视频）封装成**单个节点**，并按官方推荐的"每张脸跑一遍、链式累积缝合"自动处理多人。

This pack wraps the whole official `ComfyUI-H3-FaceRefine` pipeline (per-frame detection → face crop & magnify → low-denoise H3 latents re-sampling → per-frame denoise → feathered paste-back) into **one node**, and processes multiple people automatically via the official "one pass per face, chain the composites" approach.

---

## 这个版本解决了什么 / What this release fixes

针对你反馈的 **"修复远处小脸老是出问题"**，本版本对照技术视频（B 站 BV1L8YY6qEpi / YouTube `zmSeO6GjBBM`，作者 Benji's AI Playground）做了两项针对性优化：

This release targets exactly the reported failure — **"distant small faces always come out wrong"** — based on the technique demoed in the reference video (Bilibili BV1L8YY6qEpi / YouTube `zmSeO6GjBBM`, Benji's AI Playground):

1. **远景 person 全身兜底 / Person whole-body fallback.**
   远景里人脸太小，纯人脸检测器频繁丢帧，跟踪框会在脸外乱跳，H3 一放大就脑补出错位五官。现在节点自动调用本机 `person_yolov8m-seg.pt` 全身分割模型：人脸检测不到的帧，从人体框顶部反推头部位置，跟踪连续稳定（可关，见 `person_fallback`）。
   When the face is tiny in a long shot, the face detector drops frames constantly and the tracking box jumps off the face, so H3 hallucinates misaligned features. The node now auto-loads `person_yolov8m-seg.pt`: on frames the face detector misses, the head is located from the top of the whole-body box, giving continuous, stable tracking (toggleable via `person_fallback`).

2. **小脸 denoise 调到推荐区间 / Small-face denoise tuned.**
   远景小脸的重绘强度从 0.7 降到 **0.4**（视频作者实测 0.35–0.45 最自然），做增强式重绘而非全量脑补，避免"塑料感/独眼"。
   Small-face denoise is lowered from 0.7 to **0.4** (the author's tested sweet spot 0.35–0.45) so H3 enhances rather than fully hallucinating, killing the "plastic look / one-eye" artifacts.

---

## 安装 / Installation

把整个 `BSAI_ComfyUI_face_refine` 文件夹放进 ComfyUI 的 `custom_nodes/`，重启 ComfyUI，搜索节点 **「BSAI Face Refine」**。

Drop the whole `BSAI_ComfyUI_face_refine` folder into ComfyUI's `custom_nodes/`, restart, then search for **「BSAI Face Refine」**.

### 依赖 / Dependencies

| 依赖 Dependency | 用途 Purpose |
| --- | --- |
| `ComfyUI-H3-FaceRefine` | 人脸检测 / 裁剪 / 缝合 (face detect / crop / stitch) |
| `ComfyUI-H3-NativeAudioLock` | 口型/音频锁定，可选 (lip/audio lock, optional) |
| MiniMax H3 模型四件套 | H3 重绘 (model + CLIP + VAE + audio VAE) |
| ultralytics 人脸模型 | `models/ultralytics/bbox/face_yolov8*.pt` |
| ultralytics 全身模型 | `models/ultralytics/segm/person_yolov8m-seg.pt`（远景兜底 / long-shot fallback） |
| insightface | 多人身份跟踪 (multi-person identity) |

```bash
pip install ultralytics insightface onnxruntime-gpu scenedetect
```

---

## 快速开始 / Quick Start

拖入 `examples/BSAI_Face_Refine_example.json`，把 `VHS_LoadVideo` 的视频换成你自己的即可。

Drag in `examples/BSAI_Face_Refine_example.json` and point `VHS_LoadVideo` at your own video.

```
LoadVideo(VHS) ─▶ frames ─▶ BSAI Face Refine ─▶ images ─▶ VideoCombine(VHS)
                                 ▲
H3 Loader (model+CLIP+VAE+audioVAE) ─┘
原视频 audio ─▶ audio (可选 / optional, 锁口型 / locks lips)
```

### 两个示例 / Two example workflows

| 文件 File | 用途 Purpose |
| --- | --- |
| `examples/BSAI_Face_Refine_example.json` | 通用默认（768 画布，近/中景）default, near/mid shot |
| `examples/BSAI_Face_Refine_longshot.json` | **远景小脸专用**：512 画布 + crop_factor 3.5 + denoise 0.4 + person 兜底 on |

---

## 参数 / Parameters

| 参数 Param | 默认 Default | 说明 Description |
| --- | --- | --- |
| `detector` | face_yolov8m | 人脸检测模型；n 更快，m/c 更准 |
| `steps` | 8 | 采样步数 (official face workflow = 8) |
| `denoise` | 0.35 | 大脸重绘强度 0.25–0.45；崩坏严重 0.5–0.55 |
| `small_face_denoise` | 1.0 | 小脸初始强度（远景模式下实际被自动降到 0.4） |
| `canvas_size` | 768 | 裁剪画布；远景建议 512 |
| `confidence` | 0.35 | 检测置信度 |
| `crop_factor` | 2.5 | 人脸外扩倍数；远景建议 3.5 |
| `person_fallback` | **False** | 远景丢帧兜底，默认关。仅纯远景小脸且脸框频繁丢失时手动开；中景/夜景开了会把裁剪框拉偏导致噪点 |
| `scheduler` / `sampler_name` | beta / euler | 与官方人脸工作流一致 |
| `identity_ref_1..4` | 空 | 多人身份参考图（正面清晰单人照） |
| `stage2_enable` | False | 第二阶段细节增强（VOSR / CodeFormer 等，可选） |

---

## 远景小脸工作原理 / How long-shot small faces are handled

源脸高 < 40px 时，节点自动：
When the smallest face in the clip is < 40px tall, the node automatically:

1. **重裁 / Re-crops**：画布 768→512、crop_factor 2.5→3.5，把放大倍数从 ~9–13x 降到 ~5–7x，给 H3 足够像素合成完整五官；
2. **降 denoise / Lowers denoise**：小脸帧 denoise → **0.4**（视频实测 0.35–0.45），增强式重绘防脑补；
3. **person 兜底 / Person fallback**：人脸检测丢帧的帧，用 `person_yolov8m-seg.pt` 全身框顶部定位头部，不再盲目插值。

报告（report 输出）会打印 `[BSAIFaceRefine-v5] person fallback detector = ...` 与 `[小脸模式]` 三条。
The `report` output prints `[BSAIFaceRefine-v5] person fallback detector = ...` and three `[小脸模式]` lines.

---

## 多人 / Multi-person

在 `identity_ref_1 ~ identity_ref_4` 各接一张身份参考图，节点会逐人跑一遍并链式缝合。不接 = 自动跟踪最大脸一次。
Wire one clear frontal portrait per person into `identity_ref_1..4`; the node runs one pass per person and chains the composites. Leave empty = track the largest face once.

---

## FAQ

**没检测到脸 / No face detected?** 调低 `confidence` (0.2–0.3)，或换 `face_yolov9c.pt`。
**OOM?** 调小 `canvas_size` (512)、换 TURBO 模型、分段处理。
**脸和背景颜色不一致?** 调高 `colour_match`；贴图感则把 `blend` 降到 0.8–0.9。
**远景小脸仍崩坏?** 用 `BSAI_Face_Refine_longshot.json`；确认 `models/ultralytics/segm/person_yolov8m-seg.pt` 存在、`person_fallback=True`。

---

## 更新日志 / Changelog

- **v5 (2026-09-13)**：新增 `person_fallback`（person 全身模型兜底头部定位，解决远景小脸跟踪断裂/五官错位）；小脸 denoise 0.7→0.4（对齐视频实测 0.35–0.45）；新增远景专用示例 `BSAI_Face_Refine_longshot.json`。
- v4.1 (2026-09-13)：NestedTensor 解码修复；远景小脸自动降倍（<40px → 512 画布 + crop_factor 3.5）。
- v1.0 (2026-09-12)：首发。单人自动 + 多人链式、音频锁定、逐帧去噪，参数对齐官方人脸工作流。
