# ORB-SLAM3 对齐 YOLOv8-ORB-SLAM3 的 Mask 数据传递改动说明

更新时间：2026-04-08

## 1. 你最初需求的落地结论

目标是按 YOLOv8-ORB-SLAM3 的思路，把语义分割 Mask 从外部输入一路传到 ORB 特征提取端，并用 Mask 过滤动态区域特征点。

当前状态分两部分：

1. ORB-SLAM3 核心链路（System -> Tracking -> Frame -> ORBextractor）已经改完，并且能接收 Mask。
2. ROS2 封装节点 `semantic_rgbd_node` 里还没有真正把 Mask 传进 `TrackRGBD`，目前是预留注释状态。

## 2. 对照 YOLOv8-ORB-SLAM3 的迁移策略

YOLOv8-ORB-SLAM3 原项目把 YOLO 逻辑与 Tracking 绑定得更深（例如 `Tracking.h` 里直接包含 YOLO 头文件）。

本次迁移采用了更稳妥的工程化方案：

1. ORB 内核只“消费 mask”，不“生产 mask”。
2. YOLO 推理与消息同步放在外层（ROS2 节点层）异步处理。
3. 通过可选参数把 Mask 从接口层传入，不破坏旧调用。

这样做的好处是：

1. 保持 ORB-SLAM3 核心可维护性。
2. 不把推理耗时耦合进 Tracking 主线程。
3. 不传 Mask 时保持原版行为，兼容已有工程。

## 3. 已完成改动（核心链路）

### 3.1 System 层

1. `System::TrackRGBD(...)` 新增可选参数 `const cv::Mat &semanticMask = cv::Mat()`。
2. 在入口处做尺寸对齐（不一致时 `INTER_NEAREST` resize）。
3. 将 Mask 下传到 `Tracking::GrabImageRGBD(...)`。

涉及文件：

1. `src/ORB_SLAM3/include/System.h`
2. `src/ORB_SLAM3/src/System.cc`

### 3.2 Tracking 层

1. `Tracking::GrabImageRGBD(...)` 新增可选 `semanticMask`。
2. 构造 `Frame` 时把 `semanticMask` 继续下传。

涉及文件：

1. `src/ORB_SLAM3/include/Tracking.h`
2. `src/ORB_SLAM3/src/Tracking.cc`

### 3.3 Frame 层

1. RGB-D 构造函数新增 `semanticMask` 参数。
2. `Frame::ExtractORB(...)` 新增可选 `mask` 参数。
3. 左目/RGB-D 特征提取时把 `mask` 传给 ORBextractor。

涉及文件：

1. `src/ORB_SLAM3/include/Frame.h`
2. `src/ORB_SLAM3/src/Frame.cc`

### 3.4 ORBextractor 层

1. `ORBextractor::operator()` 中启用 `_mask` 过滤逻辑。
2. 将输入 Mask 统一到单通道 `CV_8UC1`。
3. 尺寸不一致时 resize 到当前图像大小。
4. 在每层金字塔对 keypoint 进行像素级筛选：
- `mask(y, x) > 0` 保留
- `mask(y, x) == 0` 剔除

涉及文件：

1. `src/ORB_SLAM3/include/ORBextractor.h`
2. `src/ORB_SLAM3/src/ORBextractor.cc`

## 4. 当前未完成点（你会感觉“还没全做完”的原因）

在 ROS2 节点 [src/orb_slam3_ros2/src/semantic_rgbd_node.cpp](../src/semantic_rgbd_node.cpp#L48) 内，当前仍是预留注释：

1. 没有订阅语义 Mask 话题。
2. 没有维护最新 Mask 缓冲。
3. 调用 `TrackRGBD` 时仍是三参数版本（未传 `semanticMask`）。

所以现在虽然 ORB 内核已支持 Mask，但运行路径里还未喂入真实 Mask。

## 5. 推荐接线方式（与 YOLOv8-ORB-SLAM3 目标一致）

建议在 ROS2 层实现异步融合：

1. RGB + Depth 用 ApproximateTime 同步，驱动主 SLAM 回调。
2. Mask 单独订阅，回调里只更新 `latest_mask`（加互斥锁）。
3. 主回调每帧读一份最新 Mask（允许为空），调用：

```cpp
mSLAM->TrackRGBD(rgb, depth, timestamp, {}, "", latest_mask);
```

这样即使 YOLO 帧率低于相机，也能稳定运行，不阻塞 Tracking。

## 6. 已有文档位置

你仓库里已经有一份核心实现文档：

1. [src/ORB_SLAM3/docs/semantic_mask_integration_changes.md](../../ORB_SLAM3/docs/semantic_mask_integration_changes.md)

本文档是补充版，重点回答“对照 YOLOv8-ORB-SLAM3 后目前做到哪一步、还差哪一步”。
