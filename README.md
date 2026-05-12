<h1 align="center">🎯 OpenCV Computer Vision Practice (Python)</h1>

<p align="center">
  <b>Master OpenCV from Pixels to Real-World Projects in 17 Hands-on Chapters</b><br/>
  <i>OpenCV 计算机视觉实战 — From Basics to Deep-Learning-Powered Vision</i>
</p>

<p align="center">
  <a href="./README.md">🇬🇧 English</a> •
  <a href="./README_CN.md">🇨🇳 中文</a>
</p>

<p align="center">
  <a href="https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/stargazers"><img src="https://img.shields.io/github/stars/tinyzqh/Opencv-Computer-Vision-Practice-Python-?style=flat-square&logo=github&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/network/members"><img src="https://img.shields.io/github/forks/tinyzqh/Opencv-Computer-Vision-Practice-Python-?style=flat-square&logo=github&color=blue" alt="Forks"/></a>
  <a href="https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/issues"><img src="https://img.shields.io/github/issues/tinyzqh/Opencv-Computer-Vision-Practice-Python-?style=flat-square&color=orange" alt="Issues"/></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/tests-32%2F32%20passing-brightgreen?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-curriculum">Curriculum</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-showcase">Showcase</a> •
  <a href="#-learning-path">Learning Path</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 About This Repository

A **progressive, project-driven** OpenCV learning repository. It walks you from the very basics — reading an image, thresholding, morphology — all the way to production-flavored projects: credit-card OCR, panorama stitching, parking-spot detection, exam-sheet grading, deep-learning inference with the DNN module, multi-object tracking, and drowsiness detection.

Every chapter ships with:

- ✅ **Runnable Python code** (built on `opencv-python` / `numpy`)
- ✅ **Bundled images & videos** — clone the repo and reproduce results instantly
- ✅ **A matching long-form article** (Chinese WeChat column, linked in the table below)

> 💡 **Who is this for?** CV beginners, engineers brushing up traditional vision, students preparing for CV interviews, and developers who want to combine classical OpenCV with modern deep learning.

---

## ✨ Highlights

| 🚀 17 Progressive Chapters | 🛠 9+ End-to-End Projects | 📦 Batteries Included |
| :---: | :---: | :---: |
| Smooth difficulty curve from pixel ops to DNN inference | Credit-card OCR · panorama · parking lot · OMR · drowsiness · more | Data, models and code — just `python xxx.py` |

| 📝 Article per Chapter | 🎓 Classic CV + Deep Learning | 🌱 Actively Maintained |
| :---: | :---: | :---: |
| Each chapter has a deep-dive article explaining theory & code | Harris / SIFT / Optical Flow + Caffe / dlib / DNN | Issues and PRs warmly welcomed |

---

## 📚 Curriculum

| #  | Chapter                              | Keywords                                                  | Companion Article                                                                                                | Code Folder  |
| -- | ------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------ |
| 01 | Introduction                         | Roadmap & motivation                                      | [📖 Kickoff post](https://mp.weixin.qq.com/s/hblwtPT-oC4Lsew4WUZyug)                                              | —            |
| 02 | Image Basics                         | Read image / video · ROI · channels · blending · padding  | [📖 Image Basics](https://mp.weixin.qq.com/s/mBwfLge4LaQmu37H9rNklQ)                                              | `Chapter 2`  |
| 03 | Thresholding & Smoothing             | Binarization · mean / Gaussian / median blur              | [📖 Thresholding & Smoothing](https://mp.weixin.qq.com/s/3vzdafC2Vco_eM9yzujZyg)                                  | `Chapter 3`  |
| 04 | Morphological Operations             | Erode · Dilate · Open / Close · Top-hat / Black-hat       | [📖 Morphology](https://mp.weixin.qq.com/s/8BjvINTLBq2bdkTevV62Yw)                                                | `Chapter 4`  |
| 05 | Image Gradients                      | Sobel · Scharr · Laplacian                                | [📖 Gradients](https://mp.weixin.qq.com/s/GB4_KXzUj_YlWC_1huus1g)                                                 | —            |
| 06 | Edge Detection                       | Canny full pipeline                                       | [📖 Edge Detection](https://mp.weixin.qq.com/s/gDcjwz02XwvYa0UnTs0Kgg)                                            | —            |
| 07 | Pyramids & Contours                  | Gaussian / Laplacian pyramid · contour approximation      | [📖 Pyramids & Contours](https://mp.weixin.qq.com/s/GUQ4m6FIX5yVybuBetvvxA)                                       | —            |
| 08 | Histograms & Fourier Transform       | Histogram · DFT · frequency-domain filtering              | [📖 Histograms & Fourier](https://mp.weixin.qq.com/s/N-7uHkec2C3fojw96t1O1g)                                      | —            |
| 09 | 💳 **Project 1: Credit-Card OCR**    | Template matching · morphology · perspective              | [📖 Credit-Card OCR (with code)](https://mp.weixin.qq.com/s/7GgH8_BNvJJPx6PSNx8vzA)                               | `Chapter 9`  |
| 10 | Corner Detection                     | Harris                                                    | [📖 Harris Corners](https://mp.weixin.qq.com/s/TtisJ6VFg6MAEOsYSM7amg)                                            | `Chapter 11` |
| 11 | SIFT Features                        | Scale-invariant feature transform                         | [📖 SIFT](https://mp.weixin.qq.com/s/njWAlUt3CnXwLIYEIR2yuA)                                                      | `Chapter 12` |
| 12 | 🌄 **Project 2: Panorama Stitching** | Feature matching · homography · `Stitcher`                | [📖 Panorama Stitching](https://mp.weixin.qq.com/s/2-znsJow2J6g0fgp433uVA)                                        | `Chapter 13` |
| 13 | 🅿️ **Project 3: Parking-Spot Detection** | Video processing · CNN classification · occupancy state | [📖 Parking Lot](https://mp.weixin.qq.com/s/VO76bNT3QrbOxQpm7XqpMQ)                                               | `Chapter 14` |
| 14 | 📝 **Project 4: OMR Sheet Grading**  | Perspective transform · contour sorting · bubble detection | [📖 OMR / Answer Sheet](https://mp.weixin.qq.com/s/Smd1VaIcrz31v7cJj0XTvA)                                        | `Chapter 15` |
| 15 | 🎥 **Project 5: Background Subtraction** | MOG2 · KNN · foreground extraction                     | [📖 Background Modeling](https://mp.weixin.qq.com/s/4uYal6mLbGOZebhDT2hINA)                                       | `Chapter 16` |
| 16 | 🎞 **Project 6: Optical Flow**       | Lucas-Kanade optical flow                                 | [📖 Optical Flow](https://mp.weixin.qq.com/s/kOL4X6cGyix2NGCMQgrblA)                                              | `Chapter 17` |
| 17 | 🧠 **Project 7: OpenCV DNN**         | Caffe · GoogleNet · image classification                  | [📖 OpenCV DNN](https://mp.weixin.qq.com/s/RvWT_mce0I04eAOXweAUfg)                                                | `Chapter 18` |
| 18 | 🚗 **Project 8: Multi-Object Tracking** | dlib · multithreaded tracking                          | code only                                                                                                        | `Chapter 19` |
| 19 | 😴 **Project 9: Drowsiness Detection** | Facial landmarks · EAR · blink / yawn detection         | code only                                                                                                        | `Chapter 21` |

> ⭐ Found it useful? **Drop a Star** so more people can discover it!

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-.git
cd Opencv-Computer-Vision-Practice-Python-
```

### 2. Install dependencies

We recommend an isolated environment with `conda` or `venv`:

```bash
pip install -r requirements.txt
```

That single file pulls everything you need: OpenCV (contrib build, for SIFT and legacy trackers), NumPy, Matplotlib, imutils, PyTorch + torchvision (for Chapter 14), dlib and SciPy (for Chapter 19 & 21).

<details>
<summary>Click to expand — manual install</summary>

```bash
# Core
pip install "opencv-contrib-python>=4.5,<5.0" numpy matplotlib

# Extras used by selected chapters
pip install imutils                       # Chapter 9 / 15
pip install torch torchvision             # Chapter 14 — Parking Lot (PyTorch)
pip install dlib scipy                    # Chapter 19 / 21 — Tracking & Drowsiness
```

</details>

> ✅ **Tested on Python 3.12 + OpenCV 4.13 + PyTorch 2.11 + NumPy 2.4.** All 32 example scripts run end-to-end.
> The codebase has been adapted to modern OpenCV 4.x (new `findContours` return shape, `cv2.legacy.*` trackers, strict integer coordinate types) — no version pinning gymnastics required.

### 3. Run any example

```bash
# Credit-card OCR
cd "Chapter 9"
python ocr_template_match.py --image images/credit_card_01.png --template ocr_a_reference.png

# Panorama stitching
cd "Chapter 13"
python ImageStiching.py

# Background subtraction
cd "Chapter 16"
python "back model.py"

# Parking-spot detection (Chapter 14) — two-step
cd "Chapter 14"
python train.py           # transfer-learning a VGG16 classifier (~90% val acc)
python park_test.py       # run the full detection + classification pipeline
```

---

## 🖼 Project Showcase

<table>
  <tr>
    <th>Project</th>
    <th>What You Will Learn</th>
    <th>Folder</th>
  </tr>
  <tr>
    <td>💳 Credit-Card OCR</td>
    <td>Combine template matching with morphology to solve real-world OCR</td>
    <td><code>Chapter 9</code></td>
  </tr>
  <tr>
    <td>🌄 Panorama Stitching</td>
    <td>SIFT keypoint matching + homography + image blending</td>
    <td><code>Chapter 13</code></td>
  </tr>
  <tr>
    <td>🅿️ Parking-Spot Detection</td>
    <td>Classical CV to segment ROIs + a PyTorch VGG16 transfer-learning classifier (90%+ val acc) — end-to-end</td>
    <td><code>Chapter 14</code></td>
  </tr>
  <tr>
    <td>📝 OMR Sheet Grading</td>
    <td>Perspective rectification + contour sorting + bubble recognition</td>
    <td><code>Chapter 15</code></td>
  </tr>
  <tr>
    <td>🧠 DNN Image Classification</td>
    <td>Load a Caffe model directly from OpenCV — no DL framework needed</td>
    <td><code>Chapter 18</code></td>
  </tr>
  <tr>
    <td>🚗 Multi-Object Tracking</td>
    <td>dlib correlation tracker + multithreading for real-time tracking</td>
    <td><code>Chapter 19</code></td>
  </tr>
  <tr>
    <td>😴 Drowsiness Detection</td>
    <td>68-point facial landmarks + EAR — build a tiny app that keeps you awake</td>
    <td><code>Chapter 21</code></td>
  </tr>
</table>

---

## 🗺 Learning Path

```
Basics (Ch.2~4)
        │
        ▼
Feature Engineering (Ch.5~8) ──► Histograms / Frequency / Edges / Contours
        │
        ▼
Keypoint Detection (Ch.10~11) ──► Harris / SIFT
        │
        ├──► 💳 First project: Credit-Card OCR (Ch.9)
        │
        ▼
Registration & Stitching (Ch.12) ──► 🌄 Panorama
        │
        ▼
Video & Temporal Vision (Ch.16~17) ──► 🎥 Background Subtraction / 🎞 Optical Flow
        │
        ├──► 🅿️ Parking-Spot Detection (Ch.14)
        ├──► 📝 OMR Sheet Grading (Ch.15)
        │
        ▼
Meets Deep Learning (Ch.18~21) ──► 🧠 DNN / 🚗 Multi-Tracking / 😴 Drowsiness
```

---

## 📦 Repository Structure

```
Opencv-Computer-Vision-Practice-Python-/
├── Chapter 2/   # Image basics (I/O, ROI, channels, blending, padding)
├── Chapter 3/   # Thresholding
├── Chapter 4/   # Morphology (erode / dilate / open / close / gradient / hat)
├── Chapter 9/   # 💳 Credit-Card OCR
├── Chapter 11/  # Harris corners
├── Chapter 12/  # SIFT & feature matching
├── Chapter 13/  # 🌄 Panorama stitching
├── Chapter 14/  # 🅿️ Parking-spot detection (with CNN training data)
├── Chapter 15/  # 📝 OMR sheet grading
├── Chapter 16/  # 🎥 Background subtraction
├── Chapter 17/  # 🎞 Optical flow
├── Chapter 18/  # 🧠 OpenCV DNN (Caffe / GoogleNet)
├── Chapter 19/  # 🚗 Multi-object tracking (dlib + multithreading)
├── Chapter 21/  # 😴 Drowsiness & blink detection
├── requirements.txt   # one-shot pip install
├── COMPATIBILITY.md   # OpenCV 4.x / NumPy 2 / PyTorch porting notes
├── README.md          # English (default)
└── README_CN.md       # 中文版
```

---

## 🛠 Troubleshooting

Running into a `findContours` unpack error, a missing `TrackerKCF_create`, or a Keras import in Chapter 14? See **[COMPATIBILITY.md](./COMPATIBILITY.md)** — every porting tweak (and the modern API it now uses) is documented there.

---

## 🤝 Contributing

The material is based on notes from the NetEase Cloud Classroom course [《OpenCV Computer Vision Practice》](https://study.163.com/course/courseMain.htm?courseId=1208943817&_trace_c_p_k2_=178c2b0aedfe41828e6aa2e8609882f6), reorganized and extended with comments and additional examples.

You are very welcome to contribute in any form:

- 🐞 **Found a bug or broken example?** — open an [Issue](https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/issues)
- ✨ **Want to add a new project?** — send a PR
- 📝 **Notes / typo fixes / extra explanations?** — PR the README or add a `notes/` folder
- ⭐ **Got value out of it?** — a Star is the simplest and most powerful thank-you

---

## 📜 License

- The code in this repository is released under the **MIT License** — feel free to study, modify and build on it.
- The companion WeChat articles are shared for educational purposes and do not necessarily reflect the platform's views.
- Any referenced course material remains **copyright of the original course authors**. Please contact the maintainer if you believe there is an infringement.

---

<p align="center">
  <b>If this repo helped you skip a few miles of detours, please consider giving it a ⭐ Star — so it can light the path for other learners.</b>
</p>

<p align="center">
  <sub>Made with ❤️ for everyone who loves computer vision.</sub>
</p>
