# Compatibility Notes / 兼容性说明

This repository has been adapted to **modern OpenCV 4.x and PyTorch**. Below is the full list of changes vs. the original course code (which targeted OpenCV 3.x + Keras 2.x), so you know what to expect when reading the source.

本仓库已适配 **现代 OpenCV 4.x 与 PyTorch**。下面列出相对原始课程代码（基于 OpenCV 3.x + Keras 2.x）所做的全部改动，方便阅读源码时对照参考。

> Reproducibility — every example was verified end-to-end on this stack:
> 复现环境 —— 所有示例脚本均在下列环境实测通过：
>
> | Component | Tested Version |
> | --- | --- |
> | Python | 3.12 |
> | opencv-contrib-python | 4.13.0 |
> | NumPy | 2.4 |
> | PyTorch / torchvision | 2.11 / 0.26 (CPU) |
> | dlib | 20.0 |
> | SciPy | 1.17 |
> | matplotlib | 3.10 |
> | imutils | 0.5.4 |

---

## 1. `cv2.findContours` return value (OpenCV 3 → 4)

OpenCV 3.x returned `(image, contours, hierarchy)`; OpenCV 4.x returns `(contours, hierarchy)`.

| File | Original | Patched |
| --- | --- | --- |
| `Chapter 9/ocr_template_match.py` (3 places) | `ref_, refCnts, hierarchy = cv2.findContours(...)` | `refCnts, hierarchy = cv2.findContours(...)` |
| `Chapter 15/get_answer.py` (2 places) | `cv2.findContours(...)[1]` | `cv2.findContours(...)[0]` |
| `Chapter 16/back model.py` | `im, contours, hierarchy = cv2.findContours(...)` | `contours, hierarchy = cv2.findContours(...)` |

## 2. Object trackers moved to `cv2.legacy.*` (OpenCV 4.5+)

`MultiTracker_create` and the classic single-object trackers (Boosting/MIL/KCF/TLD/MedianFlow/MOSSE/CSRT) live under `cv2.legacy` now.

```python
# Before
cv2.TrackerKCF_create()
cv2.MultiTracker_create()

# After
cv2.legacy.TrackerKCF_create()
cv2.legacy.MultiTracker_create()
```

Affects: `Chapter 19/multi-object-tracking/multi_object_tracking.py`.

## 3. Strict integer coordinates in `cv2.line` / `cv2.circle` (OpenCV 4)

OpenCV 4.x rejects float keypoint coordinates returned by `cv2.calcOpticalFlowPyrLK`. We cast them with `.astype(int)` before drawing.

Affects: `Chapter 17/optical flow estimation.py`.

## 4. `np.array(list_of_contours)` rejected by NumPy 2.x

NumPy 2 refuses to build a homogeneous array from variable-length arrays. We replaced a debug `print(np.array(refCnts).shape)` with `print(len(refCnts))`.

Affects: `Chapter 9/ocr_template_match.py`.

## 5. Chapter 14 — Keras → PyTorch rewrite

The original `train.py` used Keras 1.x / TF 1.x APIs that are unavailable in modern TensorFlow (`keras.layers.normalization.BatchNormalization`, `Model(input=..., output=...)`, `samples_per_epoch`, `nb_val_samples`, ...).

We rewrote the parking-spot classifier with **PyTorch + torchvision**:

- `train.py` — VGG16 transfer learning (first 10 conv layers frozen), 15 epochs, SGD + lightweight augmentation. Saves a checkpoint to `car1.pth` (~58 MB) containing `state_dict` + class names.
- `Parking.py` — `make_prediction` runs `model(tensor)` under `torch.no_grad()`. Accepts both uint8 and float32 inputs.
- `park_test.py` — replaces `keras_model()` with `load_torch_model()` that rebuilds the architecture and loads the checkpoint.

After 15 epochs on CPU the model reaches **~90% validation accuracy** on the bundled test set (38 empty + 126 occupied crops).

## 6. SIFT deprecation warning (harmless)

`cv2.xfeatures2d.SIFT_create()` still works in `opencv-contrib-python` but prints a DeprecationWarning. If you want clean output, swap to `cv2.SIFT_create()`. The current code keeps the old API to match the original tutorial.

---

## Quick fixes if you hit an error / 常见错误的快速修复

| Error | Fix |
| --- | --- |
| `ValueError: not enough values to unpack (expected 3, got 2)` near `findContours` | You're on OpenCV 4.x but reading the original (OpenCV 3.x) code. Pull the latest repo, or fix as in §1 above. |
| `AttributeError: module 'cv2' has no attribute 'TrackerKCF_create'` | OpenCV 4.5+ moved trackers — see §2. |
| `cv2.error: ... 'pt1'. Sequence item has a wrong type` in optical-flow code | Cast coordinates to `int` — see §3. |
| `ModuleNotFoundError: No module named 'keras'` | Chapter 14 now uses PyTorch — install via `pip install -r requirements.txt`. |
| `scipy ... numpy.dtype size changed` | `pip install -U scipy` (NumPy 2.x needs SciPy ≥ 1.13). |

---

If you find a regression or have an environment combo we missed, please [open an issue](https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/issues).
