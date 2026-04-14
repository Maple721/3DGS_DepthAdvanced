# 3DGS 自定义位姿渲染工具

## 简介

本工具用于从预训练的 3D Gaussian Splatting 模型出发，使用自定义的相机位姿（Pose）渲染 RGB 图像和深度图。特别适用于对 SAM 分割结果进行深度渲染的场景。

## 功能特点

- 从预训练模型加载 3D Gaussian 点云
- 支持自定义相机位姿（JSON 格式）
- 同时生成 RGB 图像和深度图
- 自动将黑色背景区域的深度值设为 0（适用于 SAM 分割结果）
- 支持指定渲染迭代轮数和 GPU 设备

## 文件结构

```
3DGS_DepthAdvanced/
├── render.sh                 # 主渲染脚本（Shell）
├── render_custom_pose.py     # 渲染核心逻辑（Python）
├── poses.json                # 示例位姿文件
└── output/                   # 输出目录
    └── {dataset_name}/
        ├── rgb/              # RGB 图像输出
        └── depth/            # 深度图输出
```

## 脚本与函数说明

### 1. 主入口脚本：[render.sh](render.sh)

Shell 脚本，负责解析命令行参数并调用 Python 渲染脚本。

**主要功能：**
- 解析命令行参数（`--model_path`、`--pose_file`、`--iteration`、`--gpu`）
- 设置 CUDA_VISIBLE_DEVICES 环境变量
- 创建输出目录
- 调用 [render_custom_pose.py](render_custom_pose.py) 执行实际渲染

**默认参数：**
- `GPU_ID=1`（默认使用 GPU 1）

**调用关系：**
```
render.sh
    └── python render_custom_pose.py
```

### 2. 核心渲染脚本：[render_custom_pose.py](render_custom_pose.py)

Python 脚本，包含渲染的核心逻辑。

**导入的模块与类：**
```python
import torch                              # PyTorch 深度学习框架
import json                               # JSON 文件解析
import numpy as np                        # 数值计算
from PIL import Image                     # 图像处理
from gaussian_renderer import render, GaussianModel  # 高斯渲染器和模型
import torchvision                        # PyTorch 视觉库
from utils.general_utils import safe_state  # 通用工具
from argparse import ArgumentParser        # 命令行参数解析
from arguments import ModelParams, PipelineParams, get_combined_args  # 参数配置
from scene.cameras import Camera          # 相机类
```

**主要函数：**

#### `load_poses_from_from_json(pose_file)`
**位置：** [render_custom_pose.py:31-52](render_custom_pose.py#L31-L52)

**功能：** 从 JSON 文件加载相机位姿数据

**输入：** `pose_file` - 位姿 JSON 文件路径

**输出：** 包含位姿字典的列表，每个字典包含：
- `name`: 图像名称
- `R`: 3x3 旋转矩阵
- `T`: 3x1 平移向量
- `FoVx`: 水平视场角
- `FoVy`: 垂直视场角
- `width`: 图像宽度（可选）
- `height`: 图像高度（可选）

#### `build_camera_from_pose(pose, data_device="cuda")`
**位置：** [render_custom_pose.py:55-89](render_custom_pose.py#L55-L89)

**功能：** 从位姿字典创建 Camera 对象

**输入：**
- `pose`: 位姿字典
- `data_device`: 数据设备（默认 "cuda"）

**输出：** [scene.cameras.Camera](scene/cameras.py#L19) 对象

**调用关系：**
```
build_camera_from_pose()
    └── Camera()  # 来自 scene.cameras
```

#### `render_set(model_path, output_dir, views, gaussians, pipeline, background, train_test_exp)`
**位置：** [render_custom_pose.py:92-144](render_custom_pose.py#L92-L144)

**功能：** 对一组相机视图进行渲染

**输入：**
- `model_path`: 模型路径
- `output_dir`: 输出目录
- `views`: 相机视图列表
- `gaussians`: GaussianModel 对象
- `pipeline`: 渲染管道参数
- `background`: 背景颜色
- `train_test_exp`: 是否使用训练/测试曝光

**处理流程：**
1. 创建输出目录结构（`rgb/` 和 `depth/`）
2. 对每个视图进行循环处理
3. 调用渲染函数获取 RGB 和深度图
4. 创建 mask（基于 RGB 黑色区域）
5. 应用 mask 到深度图
6. 保存 RGB 图像和深度图（.png 和 .npy 格式）

**调用关系：**
```
render_set()
    ├── render()  # 来自 gaussian_renderer
    │   └── GaussianRasterizer()  # 来自 diff_gaussian_rasterization
    ├── torchvision.utils.save_image()  # 保存 RGB
    ├── cv2 / numpy  # 创建 mask
    └── np.save()  # 保存原始深度
```

#### `render_sets(dataset, iteration, pipeline, pose_file, output_dir)`
**位置：** [render_custom_pose.py:147-179](render_custom_pose.py#L147-L179)

**功能：** 主渲染函数，协调整个渲染流程

**输入：**
- `dataset`: 数据集参数
- `iteration`: 迭代轮数（-1 表示使用最新）
- `pipeline`: 管道参数
- `pose_file`: 位姿文件路径
- `output_dir`: 输出目录

**处理流程：**
1. 初始化 GaussianModel
2. 从 PLY 文件加载预训练模型
3. 设置背景颜色
4. 从 JSON 文件加载位姿
5. 为每个位姿创建 Camera 对象
6. 调用 render_set 执行渲染

**调用关系：**
```
render_sets()
    ├── GaussianModel()  # 来自 gaussian_renderer
    ├── load_ply()  # GaussianModel 方法
    ├── load_poses_from_json()
    ├── build_camera_from_pose()
    └── render_set()
```

### 3. 支持模块说明

#### [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py)

提供高斯渲染的核心功能。

**主要类/函数：**

##### `GaussianModel`
**位置：** [scene/gaussian_model.py](scene/gaussian_model.py#L30)

**功能：** 管理 3D Gaussian 点云模型

**关键方法：**
- `__init__(sh_degree)` - 初始化模型
- `load_ply(path, use_train_test_exp)` - 从 PLY 文件加载模型（[scene/gaussian_model.py:263](scene/gaussian_model.py#L263)）
- `get_xyz` - 获取点云坐标
- `get_opacity` - 获取不透明度
- `get_features` - 获取颜色特征（SH 系数）
- `get_scaling` - 获取缩放
- `get_rotation` - 获取旋转
- `get_exposure_from_name(image_name)` - 获取曝光参数（[scene/gaussian_model.py:136](scene/gaussian_model.py#L136)）

##### `render(viewpoint_camera, pc, pipe, bg_color, ...)`
**位置：** [gaussian_renderer/__init__.py:18](gaussian_renderer/__init__.py#L18)

**功能：** 渲染场景，返回 RGB 图像和深度图

**输入：**
- `viewpoint_camera`: 相机视图
- `pc`: GaussianModel 对象
- `pipe`: 管道参数
- `bg_color`: 背景颜色
- `use_trained_exp`: 是否使用训练的曝光

**输出：** 包含以下键的字典：
- `render`: 渲染的 RGB 图像
- `depth`: 深度图
- `viewspace_points`: 视图空间点
- `visibility_filter`: 可见性过滤器
- `radii`: 高斯半径

**调用关系：**
```
render()
    ├── GaussianRasterizationSettings()  # 配置光栅化
    ├── GaussianRasterizer()  # 执行光栅化（来自 diff_gaussian_rasterization）
    ├── eval_sh()  # SH 系数评估（来自 utils.sh_utils）
    └── get_exposure_from_name()  # 获取曝光参数
```

#### [scene/cameras.py](scene/cameras.py)

提供相机相关的类。

##### `Camera`
**位置：** [scene/cameras.py:19](scene/cameras.py#L19)

**功能：** 表示单个相机视图

**关键属性：**
- `world_view_transform`: 世界视图变换矩阵
- `full_proj_transform`: 完整投影矩阵
- `camera_center`: 相机中心位置
- `FoVx`, `FoVy`: 视场角
- `image_width`, `image_height`: 图像分辨率
- `image_name`: 图像名称

**初始化依赖：**
- `getWorld2View2()` - 来自 [utils/graphics_utils.py](utils/graphics_utils.py#L38)
- `getProjectionMatrix()` - 来自 [utils/graphics_utils.py](utils/graphics_utils.py#L51)
- `PILtoTorch()` - 来自 [utils/general_utils.py](utils/general_utils.py#L21)

#### [arguments/__init__.py](arguments/__init__.py)

提供参数配置类。

##### `ModelParams`
**位置：** [arguments/__init__.py:47](arguments/__init__.py#L47)

**功能：** 管理模型加载参数

**参数：**
- `sh_degree`: 球谐函数阶数（默认 3）
- `source_path`: 源路径
- `model_path`: 模型路径
- `white_background`: 是否使用白色背景
- `train_test_exp`: 是否使用训练/测试曝光模式
- `data_device`: 数据设备（默认 "cuda"）

##### `PipelineParams`
**位置：** [arguments/__init__.py:66](arguments/__init__.py#L66)

**功能：** 管理渲染管道参数

**参数：**
- `convert_SHs_python`: 是否在 Python 中转换 SH
- `compute_cov3D_python`: 是否在 Python 中计算 3D 协方差
- `debug`: 是否开启调试模式
- `antialiasing`: 是否开启抗锯齿

#### [utils/general_utils.py](utils/general_utils.py)

提供通用工具函数。

##### `safe_state(silent)`
**位置：** [utils/general_utils.py:112](utils/general_utils.py#L112)

**功能：** 初始化系统状态，设置随机种子和设备

**操作：**
- 配置标准输出（添加时间戳）
- 设置随机种子为 0
- 设置 CUDA 设备

##### `PILtoTorch(pil_image, resolution)`
**位置：** [utils/general_utils.py:21](utils/general_utils.py#L21)

**功能：** 将 PIL 图像转换为 PyTorch 张量

**输入：**
- `pil_image`: PIL 图像对象
- `resolution`: 目标分辨率

**输出：** PyTorch 张量（C, H, W 格式）

#### [utils/graphics_utils.py](utils/graphics_utils.py)

提供图形学相关的工具函数。

##### `getWorld2View2(R, t, translate, scale)`
**位置：** [utils/graphics_utils.py:38](utils/graphics_utils.py#L38)

**功能：** 计算世界到视图的变换矩阵

**输入：**
- `R`: 旋转矩阵
- `t`: 平移向量
- `translate`: 额外平移
- `scale`: 缩放因子

**输出：** 4x4 变换矩阵

##### `getProjectionMatrix(znear, zfar, fovX, fovY)`
**位置：** [utils/graphics_utils.py:51](utils/graphics_utils.py#L51)

**功能：** 计算投影矩阵

**输入：**
- `znear`: 近裁剪面
- `zfar`: 远裁剪面
- `fovX`: 水平视场角
- `fovY`: 垂直视场角

**输出：** 4x4 投影矩阵

#### [utils/sh_utils.py](utils/sh_utils.py)

提供球谐函数（Spherical Harmonics）相关的工具。

##### `eval_sh(sh_degree, sh, dirs)`
**功能：** 评估球谐函数，将 SH 系数转换为 RGB 颜色

**输入：**
- `sh_degree`: SH 阶数
- `sh`: SH 系数
- `dirs`: 方向向量

**输出：** RGB 颜色值

#### [utils/system_utils.py](utils/system_utils.py)

提供系统相关的工具函数。

##### `mkdir_p(folder_path)`
**位置：** [utils/system_utils.py:16](utils/system_utils.py#L16)

**功能：** 递归创建目录（相当于 `mkdir -p`）

### 4. C++ 扩展模块

#### `diff_gaussian_rasterization`

**位置：** [submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py](submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py)

**功能：** 可微高斯光栅化 C++ 扩展

**关键类：**
- `GaussianRasterizationSettings`: 光栅化配置
- `GaussianRasterizer`: 执行高斯光栅化

#### `simple_knn`

**位置：** [submodules/simple-knn/](submodules/simple-knn/)

**功能：** K 近邻搜索 C++ 扩展

## 位姿文件格式

`poses.json` 文件格式如下：

```json
{
    "poses": [
        {
            "name": "00001",
            "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
            "T": [t1, t2, t3],
            "FoVx": 1.0,
            "FoVy": 0.8,
            "width": 1920,
            "height": 1080
        },
        {
            "name": "00002",
            "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
            "T": [t1, t2, t3],
            "FoVx": 1.0,
            "FoVy": 0.8,
            "width": 1920,
            "height": 1080
        }
    ]
}
```

参数说明：
- `name`: 图像名称（字符串，用于命名输出文件）
- `R`: 3x3 旋转矩阵（列表嵌套列表格式）
- `T`: 3x1 平移向量（列表格式）
- `FoVx`: 水平视场角（弧度制）
- `FoVy`: 垂直视场角（弧度制）
- `width`: 图像宽度（可选，默认 1920）
- `height`: 图像高度（可选，默认 1080）

## 使用方法

### 基本用法

```bash
cd /home/wxy/project/3DGS_DepthAdvanced

./render.sh --model_path <数据集路径> --pose_file <位姿文件路径>
```

### 完整参数

```bash
./render.sh --model_path <数据集路径> \
            --pose_file <位姿文件路径> \
            [--iteration N] \
            [--gpu ID]
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model_path` | 是 | 预训练模型路径（相对于 `output/` 目录） |
| `--pose_file` | 是 | 位姿 JSON 文件路径 |
| `--iteration` | 否 | 渲染使用的模型迭代轮数（默认使用最新轮数） |
| `--gpu` | 否 | 使用的 GPU ID（默认使用 GPU 1） |

### 使用示例

#### 示例 1：基本渲染

```bash
./render.sh --model_path 20260310_LiquidInjection1_sam --pose_file poses.json
```

输出：
- Model path: `output/20260310_LiquidInjection1_sam`
- Pose file: `poses.json`
- Output: `output/20260310_LiquidInjection1_sam/`
- RGB images: `output/20260310_LiquidInjection1_sam/rgb/`
- Depth: `output/20260310_LiquidInjection1_sam/depth/`

#### 示例 2：指定迭代轮数

```bash
./render.sh --model_path 20260310_LiquidInjection1_sam --pose'file poses.json --iteration 30000
```

#### 示例 3：指定 GPU

```bash
./render.sh --model_path 20260310_LiquidInjection1_sam --pose_file poses.json --gpu 0
```

#### 示例 4：同时指定迭代轮数和 GPU

```bash
./render.sh --model_path 20260310_LiquidInjection1_sam --pose_file poses.json --iteration 30000 --gpu 0
```

## 输出说明

### 目录结构

```
output/{dataset_name}/
├── rgb/
│   └── {pose_name}_rgb.png
└── depth/
    ├── {pose_name}_depth.png    # 归一化后的深度图（0-1）
    └── {pose_name}_depth.npy    # 原始深度值（保留深度尺度）
```

### 输出文件说明

1. **RGB 图像** (`rgb/{name}_rgb.png`)
   - 格式：PNG
   - 内容：从自定义位姿渲染的 RGB 图像
   - 分辨率：与位姿文件中定义的分辨率一致

2. **深度图（可视化）** (`depth/{name}_depth.png`)
   - 格式：PNG
   - 内容：归一化到 0-1 的深度图
   - 特点：黑色区域（背景）深度值为 0

3. **深度图（原始数据）** (`depth/{name}_depth.npy`)
   - 格式：NumPy .npy
   - 内容：原始深度值（未归一化）
   - 特点：黑色区域（背景）深度值为 0
   - 用途：可用于精确的深度分析和计算

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 解析命令行参数（render.sh）                                │
│    - 模型路径、位姿文件、迭代轮数、GPU ID                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 调用 render_custom_pose.py                                │
│    - 初始化参数解析器（ArgumentParser）                      │
│    - 加载参数配置（ModelParams, PipelineParams）              │
│    - 初始化系统状态（safe_state）                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 加载预训练模型（render_sets）                              │
│    - 初始化 GaussianModel                                    │
│    - 从 point_cloud/iteration_N/ 加载点云（load_ply）        │
│    - 设置背景颜色                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 加载自定义位姿（render_sets）                              │
│    - 读取 JSON 格式的相机位姿（load_poses_from_json）        │
│    - 为每个位姿创建 Camera 对象（build_camera_from_pose）    │
│      - 调用 getWorld2View2() 计算变换矩阵                   │
│      - 调用 getProjectionMatrix() 计算投影矩阵              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 渲染循环（render_set）                                    │
│    - 对每个位姿：                                             │
│      a. 调用 render() 渲染 RGB 图像                          │
│         - 创建 GaussianRasterizationSettings                 │
│         - 初始化 GaussianRasterizer (C++)                    │
│         - 可微高斯光栅化                                     │
│         - 应用曝光校正（如果启用）                             │
│      b. 获取深度图                                          │
│      c. 从 RGB 创建 mask（非黑色区域=1）                     │
│      d. 将 mask 应用到深度图（背景设为 0）                   │
│      e. 保存 RGB 图像（torchvision.utils.save_image）       │
│      f. 保存深度图（.png 和 .npy）                           │
└─────────────────────────────────────────────────────────────┘
```

## 函数调用层次关系

```
main (render.sh)
    └── python render_custom_pose.py
            ├── ArgumentParser
            ├── ModelParams
            ├── PipelineParams
            ├── get_combined_args
            ├── safe_state
            └── render_sets
                    ├── GaussianModel
                    │   └── load_ply
                    │       ├── PlyData.read
                    │       └── json.load (exposure.json)
                    ├── load_poses_from_json
                    │   └── json.load
                    ├── build_camera_from_pose
                    │   └── Camera
                    │       ├── PILtoTorch
                    │       ├── getWorld2View2
                    │       └── getProjectionMatrix
            └── render_set
                    └── render
                            ├── GaussianRasterizationSettings
                            ├── GaussianRasterizer (C++)
                            │   └── diff_gaussian_rasterization
                            ├── eval_sh
                            │   └── sh_utils
                            └── get_exposure_from_name
```

## 注意事项

1. **模型路径**：`--model_path` 参数指定的路径是相对于 `output/` 目录的相对路径。例如 `--model_path 20260310_LiquidInjection1_sam` 实际路径为 `output/20260310_LiquidInjection1_sam`。

2. **黑色区域处理**：脚本会自动检测 RGB 图像中的黑色区域（任意通道值 ≤ 10），并将对应深度值设为 0。这特别适用于 SAM 分割结果，可以去除背景区域的深度毛刺。

3. **GPU 内存**：如果遇到 OOM 错误，可以尝试：
   - 减少位姿文件中的图像数量
   - 使用 `--gpu` 参数指定有更多显存的 GPU
   - 降低位姿文件中的图像分辨率

4. **位姿文件路径**：`--pose_file` 参数可以是绝对路径或相对于当前工作目录的相对路径。

5. **曝光校正**：如果模型使用 train_test_exp 模式，脚本会自动加载或创建曝光参数。对于自定义位姿，如果未找到预训练的曝光，将使用单位矩阵作为默认曝光。

## 常见问题

### Q: 如何生成位姿文件？

A: 位姿文件需要包含相机的旋转矩阵 R 和平移向量 T。你可以从 COLMAP 或其他 SfM 工具导出相机位姿，然后转换为本工具要求的 JSON 格式。

### Q: 渲染结果中深度图有噪声怎么办？

A: 深度图的噪声主要来自 3D Gaussian 模型本身的特性。可以尝试：
- 使用更多迭代轮数训练的模型
- 调整渲染时的 pipeline 参数（如 antialiasing）

### Q: 可以同时渲染多个数据集吗？

A: 可以，每次运行脚本时指定不同的 `--model_path` 即可。输出会自动保存到对应的 `output/{dataset_name}/` 目录下。

### Q: 如何处理曝光参数？

A: 如果预训练模型包含 `exposure.json` 文件，脚本会自动加载。对于新相机，脚本会使用单位矩阵作为默认曝光参数。你可以在 `output/{dataset_name}/exposure.json` 中添加自定义曝光参数。

## 相关文件索引

| 文件 | 路径 |
|------|------|
| 主 Shell 脚本 | [render.sh](render.sh) |
| 核心渲染脚本 | [render_custom_pose.py](render_custom_pose.py) |
| 高斯模型 | [scene/gaussian_model.py](scene/gaussian_model.py) |
| 相机类 | [scene/cameras.py](scene/cameras.py) |
| 渲染器 | [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py) |
| 参数配置 | [arguments/__init__.py](arguments/__init__.py) |
| 通用工具 | [utils/general_utils.py](utils/general_utils.py) |
| 图形学工具 | [utils/graphics_utils.py](utils/graphics_utils.py) |
| 球谐函数工具 | [utils/sh_utils.py](utils/sh_utils.py) |
| 系统工具 | [utils/system_utils.py](utils/system_utils.py) |
