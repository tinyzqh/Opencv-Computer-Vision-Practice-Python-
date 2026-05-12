<h1 align="center">🎯 OpenCV 计算机视觉实战 (Python)</h1>

<p align="center">
  <b>从零到一，跟着十七章实战项目吃透 OpenCV</b><br/>
  <i>OpenCV Computer Vision Practice in Python — From Pixels to Real-World Projects</i>
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
  <a href="#-项目亮点">项目亮点</a> •
  <a href="#-章节目录">章节目录</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-实战项目预览">项目预览</a> •
  <a href="#-学习路线建议">学习路线</a> •
  <a href="#-贡献与交流">贡献交流</a>
</p>

---

## 📖 关于本仓库

这是一个**循序渐进、注重落地**的 OpenCV 实战学习仓库。从最基础的图像读取、阈值处理，一路推进到信用卡数字识别、全景拼接、停车场车位识别、答题卡判分、DNN 深度学习推理、多目标追踪、疲劳检测等真正能用在简历和项目里的案例。

每章配有：

- ✅ **可直接运行的 Python 代码**（基于 `opencv-python` / `numpy`）
- ✅ **配套的图片 / 视频测试素材**，clone 即可复现
- ✅ **对应的图文讲解文章**（公众号专栏，附在下方表格中）

> 💡 适合人群：计算机视觉初学者、想补全 CV 基础的算法工程师、准备秋招/春招视觉岗的同学、想把传统 CV 与深度学习结合的开发者。

---

## ✨ 项目亮点

| 🚀 17 章递进式课程 | 🛠 9+ 个完整实战项目 | 📦 开箱即用 |
| :---: | :---: | :---: |
| 从图像基本操作到 DNN 推理，难度梯度合理 | 信用卡识别 / 全景拼接 / 车位检测 / 答题卡判分 / 疲劳检测 … | 数据、模型、代码全部就位，`python xxx.py` 即可看到效果 |

| 📝 中文配套文章 | 🎓 既学传统 CV，也接深度学习 | 🌱 持续更新 |
| :---: | :---: | :---: |
| 每章配一篇公众号长文，原理 + 代码逐行讲解 | Harris / SIFT / 光流 + Caffe / dlib / DNN | 欢迎 Issue & PR，遇到坑一起踩一起填 |

---

## 📚 章节目录

| #  | 章节                         | 关键词                                       | 配套讲解                                                                                                            | 代码目录       |
| -- | -------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------- |
| 01 | 开篇导读                       | 专栏介绍 · 学习路线                              | [📖 开个视觉实战专栏](https://mp.weixin.qq.com/s/hblwtPT-oC4Lsew4WUZyug)                                                | —          |
| 02 | 图像基本操作                     | 读图/读视频 · ROI · 通道分离 · 图像融合 · 边界填充         | [📖 图像基本操作](https://mp.weixin.qq.com/s/mBwfLge4LaQmu37H9rNklQ)                                                   | `Chapter 2`  |
| 03 | 阈值与平滑处理                    | 二值化 · 均值/高斯/中值滤波                          | [📖 阈值与平滑处理](https://mp.weixin.qq.com/s/3vzdafC2Vco_eM9yzujZyg)                                                  | `Chapter 3`  |
| 04 | 图像形态学操作                    | 腐蚀 · 膨胀 · 开/闭运算 · 礼帽黑帽                    | [📖 图像形态学操作](https://mp.weixin.qq.com/s/8BjvINTLBq2bdkTevV62Yw)                                                  | `Chapter 4`  |
| 05 | 图像梯度计算                     | Sobel · Scharr · Laplacian                | [📖 图像梯度计算](https://mp.weixin.qq.com/s/GB4_KXzUj_YlWC_1huus1g)                                                   | —          |
| 06 | 边缘检测                       | Canny 全流程                                | [📖 边缘检测](https://mp.weixin.qq.com/s/gDcjwz02XwvYa0UnTs0Kgg)                                                     | —          |
| 07 | 图像金字塔与轮廓检测                 | 高斯/拉普拉斯金字塔 · 轮廓近似                         | [📖 图像金字塔与轮廓检测](https://mp.weixin.qq.com/s/GUQ4m6FIX5yVybuBetvvxA)                                               | —          |
| 08 | 直方图与傅里叶变换                  | Histogram · DFT · 频域滤波                    | [📖 直方图与傅里叶变换](https://mp.weixin.qq.com/s/N-7uHkec2C3fojw96t1O1g)                                                | —          |
| 09 | 💳 **项目一：信用卡数字识别**         | 模板匹配 · 形态学 · 透视                          | [📖 信用卡数字识别（含完整代码）](https://mp.weixin.qq.com/s/7GgH8_BNvJJPx6PSNx8vzA)                                          | `Chapter 9`  |
| 10 | 角点检测                       | Harris                                    | [📖 Harris 角点检测](https://mp.weixin.qq.com/s/TtisJ6VFg6MAEOsYSM7amg)                                              | `Chapter 11` |
| 11 | SIFT 特征                    | 尺度不变特征变换                                  | [📖 SIFT 特征](https://mp.weixin.qq.com/s/njWAlUt3CnXwLIYEIR2yuA)                                                   | `Chapter 12` |
| 12 | 🌄 **项目二：全景图像拼接**         | 特征匹配 · 单应性矩阵 · Stitcher                   | [📖 全景图像拼接](https://mp.weixin.qq.com/s/2-znsJow2J6g0fgp433uVA)                                                    | `Chapter 13` |
| 13 | 🅿️ **项目三：停车场车位识别**        | 视频处理 · CNN 分类 · 车位状态判断                    | [📖 停车场车位识别](https://mp.weixin.qq.com/s/VO76bNT3QrbOxQpm7XqpMQ)                                                  | `Chapter 14` |
| 14 | 📝 **项目四：答题卡自动判分**         | 透视变换 · 轮廓排序 · 答案识别                        | [📖 答题卡识别](https://mp.weixin.qq.com/s/Smd1VaIcrz31v7cJj0XTvA)                                                    | `Chapter 15` |
| 15 | 🎥 **项目五：背景建模**            | MOG2 · KNN · 前景提取                         | [📖 背景建模](https://mp.weixin.qq.com/s/4uYal6mLbGOZebhDT2hINA)                                                      | `Chapter 16` |
| 16 | 🎞 **项目六：光流估计**            | Lucas-Kanade 光流                          | [📖 光流估计](https://mp.weixin.qq.com/s/kOL4X6cGyix2NGCMQgrblA)                                                      | `Chapter 17` |
| 17 | 🧠 **项目七：OpenCV DNN 模型**   | Caffe · GoogleNet · 图像分类                  | [📖 OpenCV 的 DNN 模型](https://mp.weixin.qq.com/s/RvWT_mce0I04eAOXweAUfg)                                           | `Chapter 18` |
| 18 | 🚗 **项目八：多目标追踪**           | dlib · 多线程跟踪                              | 仅代码                                                                                                             | `Chapter 19` |
| 19 | 😴 **项目九：疲劳检测**            | 人脸关键点 · EAR · 眨眼/打哈欠检测                    | 仅代码                                                                                                             | `Chapter 21` |

> ⭐ 觉得有用，记得**点个 Star**，让更多同学看到这份资源～

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-.git
cd Opencv-Computer-Vision-Practice-Python-
```

### 2. 安装依赖

推荐使用 conda 或 venv 隔离环境：

```bash
pip install -r requirements.txt
```

一条命令搞定所有依赖：OpenCV（contrib 版，含 SIFT 与 legacy trackers）、NumPy、Matplotlib、imutils、PyTorch + torchvision（Chapter 14 用）、dlib 和 SciPy（Chapter 19 / 21 用）。

<details>
<summary>点击展开 — 手动安装</summary>

```bash
# 核心
pip install "opencv-contrib-python>=4.5,<5.0" numpy matplotlib

# 各章节额外依赖
pip install imutils                       # Chapter 9 / 15
pip install torch torchvision             # Chapter 14 停车场识别（PyTorch）
pip install dlib scipy                    # Chapter 19 / 21 追踪 & 疲劳检测
```

</details>

> ✅ **已在 Python 3.12 + OpenCV 4.13 + PyTorch 2.11 + NumPy 2.4 实测通过**，32 个示例脚本端到端全部跑通。
> 代码已适配现代 OpenCV 4.x（新的 `findContours` 返回值、`cv2.legacy.*` 命名空间、严格的整数坐标类型），无需锁旧版本依赖。

### 3. 运行任一示例

```bash
# 例：信用卡识别
cd "Chapter 9"
python ocr_template_match.py --image images/credit_card_01.png --template ocr_a_reference.png

# 例：全景拼接
cd "Chapter 13"
python ImageStiching.py

# 例：背景建模
cd "Chapter 16"
python "back model.py"

# 例：停车场车位识别（Chapter 14）—— 两步
cd "Chapter 14"
python train.py           # 迁移学习 VGG16 分类器（验证集 ~90%）
python park_test.py       # 跑完整的检测 + 分类流水线
```

---

## 🖼 实战项目预览

<table>
  <tr>
    <th>项目</th>
    <th>能学到什么</th>
    <th>所在章节</th>
  </tr>
  <tr>
    <td>💳 信用卡数字识别</td>
    <td>把模板匹配 + 形态学组合起来解决真实 OCR 问题</td>
    <td><code>Chapter 9</code></td>
  </tr>
  <tr>
    <td>🌄 全景图像拼接</td>
    <td>SIFT 特征点匹配 + 单应性矩阵 + 图像融合</td>
    <td><code>Chapter 13</code></td>
  </tr>
  <tr>
    <td>🅿️ 停车场车位识别</td>
    <td>传统 CV 划分 ROI + PyTorch VGG16 迁移学习分类（验证集 90%+）的端到端项目</td>
    <td><code>Chapter 14</code></td>
  </tr>
  <tr>
    <td>📝 答题卡自动判分</td>
    <td>透视变换矫正 + 轮廓排序 + 涂卡区域识别</td>
    <td><code>Chapter 15</code></td>
  </tr>
  <tr>
    <td>🧠 DNN 图像分类</td>
    <td>OpenCV 直接加载 Caffe 模型做推理，无需深度学习框架</td>
    <td><code>Chapter 18</code></td>
  </tr>
  <tr>
    <td>🚗 多目标追踪</td>
    <td>dlib correlation tracker + 多线程加速实战</td>
    <td><code>Chapter 19</code></td>
  </tr>
  <tr>
    <td>😴 疲劳检测</td>
    <td>68 点人脸关键点 + EAR 指标，做一个会"提神"的小程序</td>
    <td><code>Chapter 21</code></td>
  </tr>
</table>

---

## 🗺 学习路线建议

```
基础语法 (Ch.2~4)
        │
        ▼
特征工程 (Ch.5~8) ──► 直方图 / 频域 / 边缘 / 轮廓
        │
        ▼
关键点检测 (Ch.10~11) ──► Harris / SIFT
        │
        ├──► 💳 第一个实战：信用卡识别 (Ch.9)
        │
        ▼
图像配准与拼接 (Ch.12) ──► 🌄 全景拼接
        │
        ▼
视频与时序 (Ch.16~17) ──► 🎥 背景建模 / 🎞 光流
        │
        ├──► 🅿️ 停车场车位识别 (Ch.14)
        ├──► 📝 答题卡判分 (Ch.15)
        │
        ▼
深度学习结合 (Ch.18~21) ──► 🧠 DNN / 🚗 多目标追踪 / 😴 疲劳检测
```

---

## 📦 仓库结构

```
Opencv-Computer-Vision-Practice-Python-/
├── Chapter 2/   # 图像基本操作（读图、ROI、通道、融合、Padding）
├── Chapter 3/   # 图像阈值处理
├── Chapter 4/   # 形态学操作（腐蚀/膨胀/开闭/梯度/礼帽黑帽）
├── Chapter 9/   # 💳 信用卡数字识别
├── Chapter 11/  # Harris 角点检测
├── Chapter 12/  # SIFT 特征 & 特征匹配
├── Chapter 13/  # 🌄 全景图像拼接
├── Chapter 14/  # 🅿️ 停车场车位识别（含 CNN 训练数据）
├── Chapter 15/  # 📝 答题卡自动判分
├── Chapter 16/  # 🎥 背景建模
├── Chapter 17/  # 🎞 光流估计
├── Chapter 18/  # 🧠 OpenCV DNN（Caffe / GoogleNet）
├── Chapter 19/  # 🚗 多目标追踪（dlib + 多线程）
├── Chapter 21/  # 😴 疲劳检测 & 眨眼检测
├── requirements.txt   # 一键 pip install
├── COMPATIBILITY.md   # OpenCV 4.x / NumPy 2 / PyTorch 迁移说明
├── README.md          # English (default)
└── README_CN.md       # 中文版
```

---

## 🛠 常见问题

遇到 `findContours` 解包报错、找不到 `TrackerKCF_create`、或者 Chapter 14 import Keras 报错？看 **[COMPATIBILITY.md](./COMPATIBILITY.md)** —— 所有适配改动和现代 API 用法都在那里有详细说明。

---

## 🤝 贡献与交流

仓库内容来源于网易云课堂[《OpenCV 计算机视觉实战》](https://study.163.com/course/courseMain.htm?courseId=1208943817&_trace_c_p_k2_=178c2b0aedfe41828e6aa2e8609882f6)的学习笔记，并在此基础上做了整理、注释与扩展。

非常欢迎你以任何形式参与进来：

- 🐞 **发现 Bug / 跑不通的代码**：提一个 [Issue](https://github.com/tinyzqh/Opencv-Computer-Vision-Practice-Python-/issues)
- ✨ **想新增一个实战项目**：欢迎直接 PR
- 📝 **笔记勘误 / 文章补充**：PR 修改 README 或新增 `notes/` 即可
- ⭐ **觉得对你有帮助**：右上角点个 Star 是最大的鼓励

---

## 📜 版权与许可

- 本仓库代码采用 **MIT License** 开源，可自由学习、修改、二次开发。
- 配套的微信公众号文章用于传递和分享，不代表本平台赞同其观点或对其真实性负责。
- 引用的原始课程素材，**版权归原课程作者所有**。如有侵权请联系作者删除。

---

<p align="center">
  <b>如果这份资料帮你少走了几公里弯路，欢迎点亮右上角的 ⭐ Star，让它能照亮更多同行者。</b>
</p>

<p align="center">
  <sub>Made with ❤️ for everyone who loves computer vision.</sub>
</p>
