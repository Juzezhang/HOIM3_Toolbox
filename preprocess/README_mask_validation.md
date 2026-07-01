# Multi-View Mask Validation 使用说明

## 概述

本工具使用多视几何约束来验证多视角视频中的mask分割质量。主要用于检测mask追踪中的常见问题：
- **ID互换**: 遮挡后两个物体的mask ID发生交换
- **mask丢失**: 某一帧的mask消失或面积为零
- **错误识别**: mask覆盖了错误的物体区域

## 算法原理

### 步骤1: 构建Visual Hull (迭代加权投票, GPU批量)

1. 将所有视角的mask批量投影到3D体素空间 (`torch.bmm`)，使用可见性归一化统计占用率
2. 迭代优化: 使用体素级precision (纯tensor运算) 计算每个视角的一致性，降低不一致视角的权重
3. 经过2轮迭代后，得到可靠的3D占用体素 (visual hull)
4. 多帧共享体素网格 (union bbox)，通过 `--batch_size` 控制并行帧数

### 步骤2: BBox投影 + 重叠检查

1. 将占用体素投影到每个视角的2D平面，计算紧凑的轴对齐包围框 (bbox)
2. 对bbox施加padding (默认15%)，容纳体素离散化误差
3. 计算mask在bbox内的像素比例: `overlap = mask_pixels_in_bbox / total_mask_pixels`
4. 如果 `overlap > min_overlap` (默认0.05) 则认为mask有效

### 决策逻辑
```
valid = (mask_area > 0) AND (overlap_ratio > min_overlap)
```
- **ID互换**: mask在完全错误的位置，与投影bbox无重叠 → overlap≈0 → INVALID
- **mask丢失**: 面积为零 → 直接INVALID
- **轻微边界偏差**: mask大部分在bbox内 → overlap很高 → VALID
- **部分遮挡**: mask至少有一部分在bbox内 → overlap > 0.05 → VALID

## 数据加载流水线

脚本使用**线程prefetch流水线**，确保GPU计算与数据加载完全重叠：

```
时间轴 ──────────────────────────────────────────────►

线程池:  [加载 batch 0] [加载 batch 1]  [加载 batch 2]  ...
                        ↓              ↓              ↓
GPU:                    [处理 batch 0] [处理 batch 1] [处理 batch 2] ...
                        ↓              ↓              ↓
保存线程:               [保存 batch 0] [保存 batch 1] [保存 batch 2] ...
```

### 关键设计

1. **并行数据解压** (`--num_workers`): `ThreadPoolExecutor` 中多个线程同时解压不同帧的数据。numpy的zlib解压 (NPZ) 和 LZ4解压 (Shard) 都在C层释放GIL，线程可真正并行I/O
2. **Prefetch**: GPU处理当前batch时，线程池已在加载下一个batch的数据。GPU处理完毕后数据已就绪，无需等待
3. **异步保存**: 结果写入在后台线程完成，不阻塞GPU进入下一轮计算
4. **内存优化**: 加载时对视角切片使用 `.copy()`，仅保留所需的V个视角 (而非全部42个)，允许完整数组被GC回收
5. **格式自适应**: 通过 `--mask_format` 自动选择NPZ或Shard加载路径。Shard模式下每帧内部并行解压7个object shard

## 文件结构

```
scripts/
├── multi_view_mask_check.py        # 主验证脚本 (GPU加速)
├── visualize_mask_validity.py      # 可视化脚本
├── convert_masks_npz_to_lz4.py    # NPZ → Shard格式转换脚本
└── utils/
    ├── camera_utils.py             # 相机参数工具
    ├── mask_io.py                  # Mask I/O统一层 (NPZ + Shard格式)
    └── voxel_utils.py              # 体素工具 (独立工具库，主脚本未直接使用)
```

## Mask存储格式

### 支持的格式

脚本支持两种mask存储格式，通过 `--mask_format` 参数选择 (默认 `auto` 自动检测):

| 格式 | 存储方式 | 压缩 | 特点 |
|------|---------|------|------|
| **NPZ** (原始) | 每帧一个 `.npz` 文件 | zlib | 通用兼容，解压持有GIL |
| **Shard** (推荐) | 每object一个 `.shard` 文件 | LZ4 + bitpack | 体积更小(0.51x)，LZ4释放GIL |

### Shard格式说明

Shard格式使用 **object-major** 存储布局，每个序列仅 ~8 个文件:

```
mask_shards/
└── bedroom_data01/
    ├── meta.json           # 序列元数据 (objects, views, height, width, frame_ids)
    ├── bed.shard           # 所有帧的bed mask (LZ4+bitpack)
    ├── person0.shard
    ├── person1.shard
    ├── cushion.shard
    ├── smallsofa.shard
    ├── television.shard
    └── book.shard
```

每个 `.shard` 文件内部结构:
```
[Header 10B: magic "MSK1" + version + num_frames]
[Index: num_frames × 16B (frame_id, offset, comp_size)]
[LZ4压缩帧数据: bitpack(mask) → LZ4 compress]
```

优势:
- **体积**: bitpack (8x缩减) + LZ4 → 比NPZ小49% (22.84GB → 11.75GB/序列)
- **速度**: LZ4解压释放GIL，支持7个object并行解压
- **NFS友好**: 每序列仅8个文件 (vs NPZ的2万+个小文件)
- **随机访问**: 内置帧索引，支持O(1)跳帧读取

### 格式转换 (NPZ → Shard)

```bash
# 转换单个序列 (含验证)
python scripts/convert_masks_npz_to_lz4.py \
    --src_root "/simurgh/group/juze/datasets/HOI-M3" \
    --dst_root "/simurgh2/datasets/HOI-M3" \
    --sequences bedroom_data01 \
    --num_workers 12 --validate

# 转换全部序列 (支持resume，跳过已转换的)
python scripts/convert_masks_npz_to_lz4.py \
    --src_root "/simurgh/group/juze/datasets/HOI-M3" \
    --dst_root "/simurgh2/datasets/HOI-M3" \
    --num_workers 12 --validate
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src_root` | (必填) | 源HOI-M3根目录 (包含 `mask_npz/`) |
| `--dst_root` | (必填) | 目标根目录 (将创建 `mask_shards/`) |
| `--sequences` | 全部 | 指定要转换的序列名 |
| `--num_workers` | 12 | multiprocessing worker数 |
| `--compression_level` | 9 | LZ4压缩级别 (1-16) |
| `--validate` | false | 转换后随机抽样验证正确性 |
| `--validate_samples` | 5 | 每序列验证帧数 |

转换速度: ~4 frames/s (8 workers)，单序列(~21K帧)约需 ~90分钟

### A/B性能对比 (bedroom_data01)

| 指标 | NPZ | Shard | 变化 |
|------|-----|-------|------|
| 存储大小 | 22.84 GB | 11.75 GB | **0.51x (小49%)** |
| 10视角处理速度 | 0.55 s/帧 | 0.52 s/帧 | ~5%快 |
| 42视角处理速度 | 2.15 s/帧 | 2.00 s/帧 | ~7%快 |
| 结果一致性 | — | 完全一致 | 20帧验证通过 |

注: 在mask验证脚本中速度提升较小，因为GPU计算占主导。Shard格式的优势在训练数据加载器等I/O密集场景中更为显著。

## 快速开始

### 基本用法 (默认10视角)

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity"
```

### 使用全部42个视角

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --all_views
```

### 使用Shard格式 (推荐，需先转换)

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --mask_format shard \
    --mask_root "/simurgh2/datasets/HOI-M3/mask_shards" \
    --all_views
```

`--mask_format auto` (默认) 会自动检测: 如果 `mask_shards/{seq_name}/meta.json` 存在则用shard，否则用npz。

### 调试模式 (查看详细信息)

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --verbose \
    --start_frame 0 --end_frame 5
```

### 生成可视化视频

```bash
python scripts/visualize_mask_validity.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --validity_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity_vis" \
    --combined
```

## 参数详解

### 视角选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--views` | `0 2 5 6 7 8 10 11 14 15` | 手动指定视角列表 |
| `--all_views` | false | 使用标定文件中所有可用视角 (最多42个，覆盖 `--views`) |

`--all_views` 会自动从标定文件 (`calibration.json`) 中读取所有可用的相机视角 (限制在mask的42个通道范围内)。更多视角意味着更强的多视几何约束，但处理速度会更慢。

### Mask格式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mask_format` | `auto` | Mask存储格式: `auto`/`npz`/`shard` |
| `--mask_root` | None | Shard格式根目录 (包含 `{seq_name}/meta.json`) |

`auto` 模式检测逻辑: 优先查找 `{mask_root}/{seq_name}/meta.json`，存在则用shard，否则回退npz。

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--voxel_res` | 48 | 体素分辨率。越低越快，但精度降低 |
| `--max_iters` | 2 | 迭代次数。通常2次足够 |
| `--batch_size` | 32 | GPU并行处理帧数。增大可提高吞吐量，但需要更多显存 |
| `--num_workers` | 4 | 数据加载线程数。用于并行解压和prefetch |
| `--device` | cuda | 使用GPU加速 (cuda/cpu) |

### 迭代加权阈值 (Visual Hull构建)

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--thresh_init` | 0.5 | 0.3-0.8 | 首轮占用阈值 |
| `--thresh_expand` | 0.35 | 0.2-0.6 | 后续轮占用阈值 |
| `--prec_penalty` | 0.3 | 0.2-0.6 | 迭代中权重惩罚阈值 |
| `--area_penalty` | 0.05 | 0.01-0.2 | 面积过小惩罚阈值 |

### 最终决策阈值 (BBox重叠检查)

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--bbox_padding` | 0.15 | 0.0-0.5 | 投影bbox的padding比例。**增大 → 更宽松** |
| `--min_overlap` | 0.05 | 0.01-0.3 | mask与bbox最小重叠比例。**降低 → 更宽松** |

`overlap_ratio = mask_pixels_in_bbox / total_mask_pixels`。例如0.05表示只要mask有5%的像素落在投影bbox内即认为有效。

## 参数调整指南

### 问题: 误判率太高 (正确mask被标为invalid)

**解决方案**: 增大bbox_padding + 降低min_overlap

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --bbox_padding 0.3 \
    --min_overlap 0.02
```

### 问题: 漏检率太高 (错误mask没被检测出)

**解决方案**: 减小bbox_padding + 提高min_overlap

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --bbox_padding 0.05 \
    --min_overlap 0.15
```

### 问题: 运行速度太慢

```bash
# 方法1: 增大batch_size (利用更多GPU显存)
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --batch_size 64

# 方法2: 增加数据加载线程 (I/O密集场景)
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --num_workers 8

# 方法3: 降低体素分辨率 (牺牲精度换速度)
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --voxel_res 32 \
    --max_iters 1
```

**速度优化历史**:

| 版本 | 每帧时间 (10视角) | 加速比 |
|------|-------------------|--------|
| 原始 (逐视角逐帧) | ~4.0秒/帧 | 1x |
| GPU批量 (无prefetch) | ~1.8秒/帧 | 2.2x |
| GPU批量 + prefetch + NPZ | ~0.55秒/帧 | 7.3x |
| GPU批量 + prefetch + Shard | ~0.52秒/帧 | 7.7x |

```bash
# 方法4: 使用Shard格式 (I/O更快，需先转换)
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --mask_format shard --mask_root "/simurgh2/datasets/HOI-M3/mask_shards"
```

## 推荐配置

### 默认模式 (10视角)

直接运行，无需额外参数。当前默认参数已针对HOI-M3数据集优化：

```bash
python scripts/multi_view_mask_check.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity"
```

**测试结果** (bedroom_data01, frame 0):

| Object | 10 views |
|--------|----------|
| smallsofa | 10/10 |
| bed | 10/10 |
| television | 10/10 |
| person1 | 10/10 |
| person0 | 8/10 |
| cushion | 7/10 |
| book | 6/10 |

## 输出格式

### mask_validity 目录结构

```
mask_validity/
└── bedroom_data01/
    ├── 000000.npz
    ├── 000001.npz
    └── ...
```

### NPZ文件内容

```python
import numpy as np
data = np.load('mask_validity/bedroom_data01/000000.npz')

# 每个object都有对应的validity数组
# 数组长度 = 使用的视角数量 (默认10, --all_views时为42)
# 0 = invalid, 1 = valid
print(data['person0_validity'])  # 长度为10或42的数组
print(data['bed_validity'])
```

### 视角索引对应关系

validity数组按 `--views` 参数的顺序排列：

**默认10视角** (`--views 0 2 5 6 7 8 10 11 14 15`):
- index 0 → view 0
- index 1 → view 2
- index 2 → view 5
- ...以此类推

**全部42视角** (`--all_views`):
- index 0 → view 0
- index 1 → view 1
- ...
- index 41 → view 41

## verbose模式输出解读

使用 `--verbose` 时，每个物体每个视角会打印：

```
view 5: overlap=0.950 bbox=(1177,410,1414,577) mask_area=13349 -> VALID
```

| 字段 | 含义 | 判断标准 |
|------|------|---------|
| `overlap` | mask在投影bbox内的像素比例 | > min_overlap (0.05) 则VALID |
| `bbox` | 投影bbox坐标 (x1,y1,x2,y2) | 从visual hull投影得到 |
| `mask_area` | mask总像素数 | > 100 为基本条件 |

**结果解读**:
- `overlap=1.000` → mask完全在投影bbox内，高置信度VALID
- `overlap=0.500` → mask有一半在bbox内，仍VALID (可能是部分遮挡)
- `overlap=0.067` → mask只有很小部分在bbox内，刚过阈值VALID (边缘情况)
- `overlap=0.000` → mask完全不在bbox内，高置信度INVALID (很可能是ID互换)
- `area=0` → mask为空，直接INVALID

## 可视化说明

### 颜色含义

- **绿色**: valid mask (通过验证)
- **红色**: invalid mask (未通过验证)

### 输出视频

```bash
# 单视角视频 (默认10视角)
python scripts/visualize_mask_validity.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --validity_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity_vis"
# 输出: person0_view0.mp4, person0_view2.mp4, ...

# 组合多视角视频 (推荐)
python scripts/visualize_mask_validity.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --validity_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity_vis" \
    --combined
# 输出: person0_combined.mp4

# 全部42视角的组合视频
python scripts/visualize_mask_validity.py \
    --root_path "/simurgh/group/juze/datasets/HOI-M3" \
    --seq_name "bedroom_data01" \
    --validity_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
    --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity_vis" \
    --all_views --combined
# 输出: person0_combined.mp4 (7列x6行网格, 每视角256x144)
```

可视化脚本会根据视角数量自动调整网格布局：
- 10视角: 5列x2行, 每视角384x216
- 11-21视角: 7列, 每视角320x180
- 22-42视角: 7列, 每视角256x144

## 性能

### 验证脚本 (multi_view_mask_check.py)

当前版本使用GPU批量处理 + 线程prefetch流水线。数据加载与GPU计算完全重叠：加载batch N+1的同时GPU处理batch N。

| 配置 | 每帧处理时间 | 说明 |
|------|-------------|------|
| 10视角, batch_size=32, GPU (默认) | ~0.6秒/帧 | 稳态~0.4秒/帧，首批0.8秒/帧 |
| 10视角, voxel_res=32, GPU | ~0.3-0.5秒/帧 | 更低分辨率 |

注意: 每帧时间与场景中物体数量成正比。启动时会打印GPU显存估算值，可据此调整 `--batch_size`。`--num_workers` 控制数据加载并行度，默认4线程。

### 可视化脚本 (visualize_mask_validity.py)

视频帧采用**预加载模式** (sequential read)，所有视角的视频帧在开始时一次性顺序读取并缓存，避免了逐帧随机seek的性能问题。

| 模式 | 速度 |
|------|------|
| combined, 10视角 | ~1.8秒/帧/物体 |
| combined, 42视角 | ~1.8秒/帧/物体 (帧已预加载) |

### GPU显存估算

启动时自动打印显存估算。主要占用来自mask数据上传到GPU：

| 配置 | 估算显存 |
|------|----------|
| 10视角, batch_size=32 | ~0.3 GB |
| 42视角, batch_size=32 | ~12 GB |
| 42视角, batch_size=64 | ~23 GB |

## 依赖

- numpy
- torch (GPU批量运算)
- lz4 (Shard格式LZ4解压)
- opencv-python (可视化脚本)
- tqdm (进度条)
