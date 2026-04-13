#!/usr/bin/env python3
import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


@dataclass
class TimedPose:
    t: float
    xyz: np.ndarray


class SlamEvalNode(Node):
    def __init__(self) -> None:
        super().__init__('slam_eval_node')

        self.declare_parameter('est_pose_topic', '/orb_slam3/camera_pose')
        self.declare_parameter('gt_source', 'tf')
        self.declare_parameter('gt_file', '')
        self.declare_parameter('gt_tf_topic', '/tf')
        self.declare_parameter('gt_parent_frame', '/world')
        self.declare_parameter('gt_child_frame', '/kinect')
        self.declare_parameter('time_tolerance', 0.02)
        self.declare_parameter('sync_mode', 'interp')
        self.declare_parameter('strict_sync', True)
        self.declare_parameter('max_interp_gap', 0.05)
        self.declare_parameter('min_matches_for_metrics', 20)
        self.declare_parameter('report_interval_sec', 1.0)
        self.declare_parameter('publish_paths', True)
        self.declare_parameter('publish_synced_pose', True)
        self.declare_parameter('publish_aligned_path', True)
        self.declare_parameter('align_window_size', 200)
        self.declare_parameter('min_matches_for_align', 20)

        self.est_pose_topic = self.get_parameter('est_pose_topic').value
        self.gt_source = str(self.get_parameter('gt_source').value).strip().lower()
        self.gt_file = self.get_parameter('gt_file').value
        self.gt_tf_topic = self.get_parameter('gt_tf_topic').value
        self.gt_parent_frame = self._norm_frame(self.get_parameter('gt_parent_frame').value)
        self.gt_child_frame = self._norm_frame(self.get_parameter('gt_child_frame').value)
        self.time_tolerance = float(self.get_parameter('time_tolerance').value)
        self.sync_mode = str(self.get_parameter('sync_mode').value).strip().lower()
        self.strict_sync = bool(self.get_parameter('strict_sync').value)
        self.max_interp_gap = float(self.get_parameter('max_interp_gap').value)
        self.min_matches_for_metrics = int(self.get_parameter('min_matches_for_metrics').value)
        self.report_interval = float(self.get_parameter('report_interval_sec').value)
        self.publish_paths = bool(self.get_parameter('publish_paths').value)
        self.publish_synced_pose = bool(self.get_parameter('publish_synced_pose').value)
        self.publish_aligned_path = bool(self.get_parameter('publish_aligned_path').value)
        self.align_window_size = int(self.get_parameter('align_window_size').value)
        self.min_matches_for_align = int(self.get_parameter('min_matches_for_align').value)

        if self.sync_mode not in ('nearest', 'interp'):
            self.get_logger().warn(f'Unknown sync_mode={self.sync_mode}, fallback to interp')
            self.sync_mode = 'interp'
        if self.gt_source not in ('file', 'tf'):
            self.get_logger().warn(f'Unknown gt_source={self.gt_source}, fallback to tf')
            self.gt_source = 'tf'

        self.gt_stamps: List[float] = []
        self.gt_rel_stamps: List[float] = []
        self.gt_xyz: List[np.ndarray] = []
        self._gt_tf_msg_count = 0
        self._gt_tf_match_count = 0

        if self.gt_source == 'file':
            self._load_groundtruth(self.gt_file)
        else:
            self.create_subscription(TFMessage, self.gt_tf_topic, self.gt_tf_cb, 200)

        self.matched_est_xyz: List[np.ndarray] = []
        self.matched_gt_xyz: List[np.ndarray] = []
        self.matched_est_t: List[float] = []
        self._recv_count = 0
        self._last_recv_time: Optional[float] = None
        self._first_est_time: Optional[float] = None
        self._fps = 0.0
        self._drop_unsynced = 0

        self.gt_path_pub = self.create_publisher(RosPath, '/slam_eval/groundtruth_path', 10)
        self.est_aligned_path_pub = self.create_publisher(RosPath, '/slam_eval/estimated_path_aligned', 10)
        self.metrics_pub = self.create_publisher(String, '/slam_eval/metrics', 10)
        self.gt_sync_pose_pub = self.create_publisher(PoseStamped, '/slam_eval/synced_gt_pose', 20)
        self.est_sync_aligned_pose_pub = self.create_publisher(PoseStamped, '/slam_eval/synced_est_pose_aligned', 20)

        self.est_aligned_path_msg = RosPath()
        self.gt_path_msg = RosPath()
        self.est_aligned_path_msg.header.frame_id = 'world'
        self.gt_path_msg.header.frame_id = 'world'

        self.create_subscription(PoseStamped, self.est_pose_topic, self.pose_cb, 100)
        self.create_timer(self.report_interval, self.report_metrics)

        if self.gt_source == 'file':
            self.get_logger().info(
                f'SLAM eval started. est_pose_topic={self.est_pose_topic}, gt_source=file, gt_file={self.gt_file}, '
                f'sync_mode={self.sync_mode}, strict_sync={self.strict_sync}, '
                f'tolerance={self.time_tolerance}s, max_interp_gap={self.max_interp_gap}s'
            )
        else:
            self.get_logger().info(
                f'SLAM eval started. est_pose_topic={self.est_pose_topic}, gt_source=tf, '
                f'gt_tf_topic={self.gt_tf_topic}, pair={self.gt_parent_frame}->{self.gt_child_frame}, '
                f'sync_mode={self.sync_mode}, strict_sync={self.strict_sync}, '
                f'tolerance={self.time_tolerance}s, max_interp_gap={self.max_interp_gap}s'
            )

    @staticmethod
    def _norm_frame(name: str) -> str:
        return str(name).lstrip('/')

    def _load_groundtruth(self, gt_file: str) -> None:
        gt_path = Path(gt_file)
        if not gt_path.exists():
            raise FileNotFoundError(f'groundtruth file not found: {gt_file}')

        with gt_path.open('r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                if len(parts) < 4:
                    continue
                t = float(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                self.gt_stamps.append(t)
                self.gt_xyz.append(xyz)

        if len(self.gt_stamps) < 2:
            raise RuntimeError(f'groundtruth file has too few entries: {gt_file}')

        t0 = self.gt_stamps[0]
        self.gt_rel_stamps = [t - t0 for t in self.gt_stamps]

    def pose_cb(self, msg: PoseStamped) -> None:
        self._recv_count += 1
        ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        if self._last_recv_time is not None:
            dt = ts - self._last_recv_time
            if dt > 1e-6:
                self._fps = 1.0 / dt
        self._last_recv_time = ts
        if self._first_est_time is None:
            self._first_est_time = ts

        if self.gt_source == 'file':
            ts_query = ts - self._first_est_time
        else:
            ts_query = ts

        xyz_est = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )

        gt = self.match_gt(ts_query)
        if gt is None:
            self._drop_unsynced += 1
            return

        _, gt_xyz = gt
        self.matched_est_xyz.append(xyz_est)
        self.matched_gt_xyz.append(gt_xyz)
        self.matched_est_t.append(ts)

        est_aligned_xyz = self._compute_aligned_point(xyz_est)

        if self.publish_paths:
            if self.publish_aligned_path and est_aligned_xyz is not None:
                est_pose_aligned = PoseStamped()
                est_pose_aligned.header = msg.header
                est_pose_aligned.header.frame_id = 'world'
                est_pose_aligned.pose.position.x = float(est_aligned_xyz[0])
                est_pose_aligned.pose.position.y = float(est_aligned_xyz[1])
                est_pose_aligned.pose.position.z = float(est_aligned_xyz[2])
                est_pose_aligned.pose.orientation.w = 1.0
                self.est_aligned_path_msg.header.stamp = msg.header.stamp
                self.est_aligned_path_msg.poses.append(est_pose_aligned)
                self.est_aligned_path_pub.publish(self.est_aligned_path_msg)

            gt_pose = PoseStamped()
            gt_pose.header = msg.header
            gt_pose.header.frame_id = 'world'
            gt_pose.pose.position.x = float(gt_xyz[0])
            gt_pose.pose.position.y = float(gt_xyz[1])
            gt_pose.pose.position.z = float(gt_xyz[2])
            gt_pose.pose.orientation.w = 1.0
            self.gt_path_msg.header.stamp = msg.header.stamp
            self.gt_path_msg.poses.append(gt_pose)
            self.gt_path_pub.publish(self.gt_path_msg)

        if self.publish_synced_pose:
            if self.publish_aligned_path and est_aligned_xyz is not None:
                est_sync_aligned = PoseStamped()
                est_sync_aligned.header = msg.header
                est_sync_aligned.header.frame_id = 'world'
                est_sync_aligned.pose.position.x = float(est_aligned_xyz[0])
                est_sync_aligned.pose.position.y = float(est_aligned_xyz[1])
                est_sync_aligned.pose.position.z = float(est_aligned_xyz[2])
                est_sync_aligned.pose.orientation.w = 1.0
                self.est_sync_aligned_pose_pub.publish(est_sync_aligned)

            gt_sync = PoseStamped()
            gt_sync.header = msg.header
            gt_sync.header.frame_id = 'world'
            gt_sync.pose.position.x = float(gt_xyz[0])
            gt_sync.pose.position.y = float(gt_xyz[1])
            gt_sync.pose.position.z = float(gt_xyz[2])
            gt_sync.pose.orientation.w = 1.0
            self.gt_sync_pose_pub.publish(gt_sync)

    def gt_tf_cb(self, msg: TFMessage) -> None:
        self._gt_tf_msg_count += 1
        for tr in msg.transforms:
            if self._norm_frame(tr.header.frame_id) != self.gt_parent_frame:
                continue
            if self._norm_frame(tr.child_frame_id) != self.gt_child_frame:
                continue

            t = float(tr.header.stamp.sec) + float(tr.header.stamp.nanosec) * 1e-9
            xyz = np.array([
                tr.transform.translation.x,
                tr.transform.translation.y,
                tr.transform.translation.z,
            ], dtype=np.float64)

            i = bisect.bisect_left(self.gt_stamps, t)
            if i < len(self.gt_stamps) and abs(self.gt_stamps[i] - t) < 1e-9:
                self.gt_xyz[i] = xyz
                continue

            self.gt_stamps.insert(i, t)
            self.gt_xyz.insert(i, xyz)
            self._gt_tf_match_count += 1

        # Keep memory bounded for long runs.
        if len(self.gt_stamps) > 20000:
            keep = 15000
            self.gt_stamps = self.gt_stamps[-keep:]
            self.gt_xyz = self.gt_xyz[-keep:]

    def _nearest_gt(self, t_rel: float) -> Optional[Tuple[float, np.ndarray]]:
        stamps = self.gt_rel_stamps if self.gt_source == 'file' else self.gt_stamps
        xyzs = self.gt_xyz
        if len(stamps) < 2:
            return None

        i = bisect.bisect_left(stamps, t_rel)
        cand = []
        if i < len(stamps):
            cand.append(i)
        if i > 0:
            cand.append(i - 1)
        if not cand:
            return None

        best_i = min(cand, key=lambda idx: abs(stamps[idx] - t_rel))
        dt = abs(stamps[best_i] - t_rel)
        if self.strict_sync and dt > self.time_tolerance:
            return None
        return stamps[best_i], xyzs[best_i]

    def _interp_gt(self, t_rel: float) -> Optional[Tuple[float, np.ndarray]]:
        stamps = self.gt_rel_stamps if self.gt_source == 'file' else self.gt_stamps
        xyzs = self.gt_xyz
        if len(stamps) < 2:
            return None

        i = bisect.bisect_left(stamps, t_rel)
        if i <= 0 or i >= len(stamps):
            return None

        t0 = stamps[i - 1]
        t1 = stamps[i]
        p0 = xyzs[i - 1]
        p1 = xyzs[i]

        gap = t1 - t0
        if gap <= 1e-9:
            return None

        if self.strict_sync and gap > self.max_interp_gap:
            return None

        alpha = (t_rel - t0) / gap
        alpha = min(1.0, max(0.0, alpha))
        xyz = (1.0 - alpha) * p0 + alpha * p1
        return t_rel, xyz

    def match_gt(self, t_rel: float) -> Optional[Tuple[float, np.ndarray]]:
        if self.sync_mode == 'nearest':
            return self._nearest_gt(t_rel)
        return self._interp_gt(t_rel)

    def _compute_aligned_point(self, xyz_est: np.ndarray) -> Optional[np.ndarray]:
        n = len(self.matched_est_xyz)
        if n < self.min_matches_for_align:
            return None

        if self.align_window_size > 0 and n > self.align_window_size:
            est = np.vstack(self.matched_est_xyz[-self.align_window_size:])
            gt = np.vstack(self.matched_gt_xyz[-self.align_window_size:])
        else:
            est = np.vstack(self.matched_est_xyz)
            gt = np.vstack(self.matched_gt_xyz)

        r, t = self.horn_align(est, gt)
        return (r @ xyz_est) + t

    @staticmethod
    def horn_align(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        est_centered = est_xyz - np.mean(est_xyz, axis=0)
        gt_centered = gt_xyz - np.mean(gt_xyz, axis=0)
        w = est_centered.T @ gt_centered
        u, _, vh = np.linalg.svd(w)
        r = vh.T @ u.T
        if np.linalg.det(r) < 0:
            vh[2, :] *= -1.0
            r = vh.T @ u.T
        t = np.mean(gt_xyz, axis=0) - (r @ np.mean(est_xyz, axis=0))
        return r, t

    def report_metrics(self) -> None:
        n = len(self.matched_est_xyz)
        if n < self.min_matches_for_metrics:
            if self.gt_source == 'tf':
                self.get_logger().info(
                    f'Waiting for enough matches: {n} pairs, dropped_unsynced={self._drop_unsynced}, '
                    f'tf_msgs={self._gt_tf_msg_count}, tf_matched={self._gt_tf_match_count}'
                )
            else:
                self.get_logger().info(
                    f'Waiting for enough matches: {n} pairs, dropped_unsynced={self._drop_unsynced}'
                )
            return

        est = np.vstack(self.matched_est_xyz)
        gt = np.vstack(self.matched_gt_xyz)

        r, t = self.horn_align(est, gt)
        est_aligned = (r @ est.T).T + t

        ate = np.linalg.norm(est_aligned - gt, axis=1)
        ate_rmse = float(math.sqrt(np.mean(np.square(ate))))
        ate_mean = float(np.mean(ate))

        if est_aligned.shape[0] >= 2:
            de = est_aligned[1:] - est_aligned[:-1]
            dg = gt[1:] - gt[:-1]
            rpe = np.linalg.norm(de - dg, axis=1)
            rpe_rmse = float(math.sqrt(np.mean(np.square(rpe))))
        else:
            rpe_rmse = float('nan')

        metrics_text = (
            f'fps={self._fps:.2f}, matched={n}, ATE_RMSE={ate_rmse:.4f} m, '
            f'ATE_MEAN={ate_mean:.4f} m, RPE_RMSE={rpe_rmse:.4f} m, dropped_unsynced={self._drop_unsynced}'
        )
        self.metrics_pub.publish(String(data=metrics_text))
        self.get_logger().info(metrics_text)


def main() -> None:
    rclpy.init()
    node = SlamEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
