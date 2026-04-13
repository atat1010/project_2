# 语义SLAM实时轨迹可视化指南

## 1. 目标

本文档说明如何直接读取 rosbag 发布的 /tf 作为真实轨迹，
实时可视化估计轨迹与真实轨迹，不依赖 GT 文件。

---

## 2. 统一环境与变量（先执行一次）

```bash
source /opt/ros/humble/setup.bash
cd /home/qm/semantic_slam_ws
source install/setup.bash

export BAG_PATH=/home/qm/semantic_slam_ws/src/ros2bag/freiburg3_walking/ros2_walking_halfsphere/ros2_walking_halfsphere.mcap
export RUN_NAME=yolo_on
export OUT_DIR=/home/qm/semantic_slam_ws/results/bags/ros2_walking_halfsphere_${RUN_NAME}

mkdir -p ${OUT_DIR}
```

说明：
- 做 A/B 对照时，只改 `RUN_NAME`（例如 `yolo_off` 和 `yolo_on`），避免输出互相覆盖。

---

## 3. 实时可视化（直接使用 /tf 真值）

### 3.1 不开 YOLO

不开 YOLO：

```bash
source /opt/ros/humble/setup.bash
cd /home/qm/semantic_slam_ws
source install/setup.bash

ros2 launch orb_slam3_ros2 semantic_slam.launch.py \
  play_bag:=true \
  run_yolo:=false \
  run_eval:=true \
  use_realsense:=false \
  use_viewer:=false \
  shutdown_when_bag_done:=true \
  bag_path:=${BAG_PATH} \
  est_traj_path:=${OUT_DIR}/estimated_tum.txt \
  eval_gt_source:=tf \
  eval_gt_tf_topic:=/tf \
  eval_gt_parent_frame:=/world \
  eval_gt_child_frame:=/kinect \
  eval_sync_mode:=interp \
  eval_strict_sync:=true \
  eval_max_interp_gap:=0.05 \
  eval_time_tolerance:=0.02
```

### 3.2 开 YOLO

```bash
source /opt/ros/humble/setup.bash
cd /home/qm/semantic_slam_ws
source install/setup.bash

ros2 launch orb_slam3_ros2 semantic_slam.launch.py \
  play_bag:=true \
  run_yolo:=true \
  run_eval:=true \
  use_realsense:=false \
  use_viewer:=false \
  shutdown_when_bag_done:=true \
  bag_path:=${BAG_PATH} \
  est_traj_path:=${OUT_DIR}/estimated_tum.txt \
  eval_gt_source:=tf \
  eval_gt_tf_topic:=/tf \
  eval_gt_parent_frame:=/world \
  eval_gt_child_frame:=/kinect \
  eval_sync_mode:=interp \
  eval_strict_sync:=true \
  eval_max_interp_gap:=0.05 \
  eval_time_tolerance:=0.02
```

可选参数（YOLO与SLAM时间对齐）：

```bash
mask_sync_tolerance:=0.06
```

## 4. RViz 可视化话题

实时对比主要话题：
1. `/slam_eval/estimated_path`
2. `/slam_eval/groundtruth_path`
3. `/slam_eval/synced_est_pose`
4. `/slam_eval/synced_gt_pose`
5. `/slam_eval/metrics`

---

## 5. 日志检查（确认 YOLO 真的在用）

运行 Step 2（run_yolo:=true）时，关注两类日志：

1. YOLO 节点：
- `YOLO FPS(...): ... | avg infer: ... ms | dropped(busy): ...`

2. SLAM 节点：
- `SLAM FPS(1s): ... | YOLO expected: true | YOLO alive: true/false | mask FPS(1s): ... | mask used: ...%`

判读：
- `YOLO expected: true` 且 `YOLO alive: true` 才表示 YOLO 掩码在持续参与。
- 如果长期 `YOLO alive: false`，说明 YOLO未正常产出 mask（未运行、话题不对、或推理过慢）。

---

## 6. A/B 对照最简做法

1. 先跑 yolo_off：

```bash
export RUN_NAME=yolo_off
export OUT_DIR=/home/qm/semantic_slam_ws/results/bags/ros2_walking_halfsphere_${RUN_NAME}
mkdir -p ${OUT_DIR}
```

按 3.1 执行。

2. 再跑 yolo_on：

```bash
export RUN_NAME=yolo_on
export OUT_DIR=/home/qm/semantic_slam_ws/results/bags/ros2_walking_halfsphere_${RUN_NAME}
mkdir -p ${OUT_DIR}
```

按 3.2 执行。

3. 最后比对结果文件，不会互相覆盖。
