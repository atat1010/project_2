#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class SemanticPointCloudBuilder : public rclcpp::Node {
public:
  SemanticPointCloudBuilder() : Node("semantic_pointcloud_builder") {
    // 内参直接由 semantic_pointcloud_builder.yaml 提供。
    fx_ = this->declare_parameter<double>("fx", 535.4);
    fy_ = this->declare_parameter<double>("fy", 539.2);
    cx_ = this->declare_parameter<double>("cx", 320.1);
    cy_ = this->declare_parameter<double>("cy", 247.6);
    RCLCPP_INFO(this->get_logger(), "Camera intrinsics: fx=%.4f fy=%.4f cx=%.4f cy=%.4f", fx_, fy_,
                cx_, cy_);

    voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.05);
    depth_min_ = this->declare_parameter<double>("depth_min", 0.1);
    depth_max_ = this->declare_parameter<double>("depth_max", 8.0);

    const auto rgb_topic = this->declare_parameter<std::string>("rgb_topic", "/camera/rgb/image_color");
    const auto depth_topic = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image");
    const auto mask_topic = this->declare_parameter<std::string>("mask_topic", "/yolo/mask");
    const auto pose_topic = this->declare_parameter<std::string>("pose_topic", "/orb_slam3/camera_pose");
    const auto output_topic = this->declare_parameter<std::string>("output_topic", "/semantic_global_map");

    global_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    global_cloud_->reserve(200000);

    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 5);

    rgb_sub_.subscribe(this, rgb_topic);
    depth_sub_.subscribe(this, depth_topic);
    mask_sub_.subscribe(this, mask_topic);
    pose_sub_.subscribe(this, pose_topic);

    sync_ = std::make_shared<Synchronizer>(SyncPolicy(20), rgb_sub_, depth_sub_, mask_sub_, pose_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.08));
    sync_->registerCallback(
      std::bind(&SemanticPointCloudBuilder::sync_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    RCLCPP_INFO(this->get_logger(),
                "SemanticPointCloudBuilder started. topics: rgb=%s depth=%s mask=%s pose=%s out=%s",
                rgb_topic.c_str(), depth_topic.c_str(), mask_topic.c_str(), pose_topic.c_str(),
                output_topic.c_str());
  }

private:
  using ImageMsg = sensor_msgs::msg::Image;
  using PoseMsg = geometry_msgs::msg::PoseStamped;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, PoseMsg>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void sync_callback(const ImageMsg::ConstSharedPtr &rgb_msg, const ImageMsg::ConstSharedPtr &depth_msg,
                     const ImageMsg::ConstSharedPtr &mask_msg,
                     const PoseMsg::ConstSharedPtr &pose_msg) {
    cv_bridge::CvImageConstPtr rgb_ptr;
    cv_bridge::CvImageConstPtr depth_ptr;
    cv_bridge::CvImageConstPtr mask_ptr;

    try {
      rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
      depth_ptr = cv_bridge::toCvShare(depth_msg, "32FC1");
      mask_ptr = cv_bridge::toCvShare(mask_msg, "mono8");
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
      return;
    }

    const cv::Mat &rgb = rgb_ptr->image;
    const cv::Mat &depth = depth_ptr->image;
    const cv::Mat &mask = mask_ptr->image;

    if (rgb.empty() || depth.empty() || mask.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Received empty frame(s), skip this sync callback");
      return;
    }

    if (rgb.rows != depth.rows || rgb.cols != depth.cols || rgb.rows != mask.rows ||
        rgb.cols != mask.cols) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Frame size mismatch: rgb(%dx%d), depth(%dx%d), mask(%dx%d)", rgb.cols,
                           rgb.rows, depth.cols, depth.rows, mask.cols, mask.rows);
      return;
    }

    auto local_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    local_cloud->points.reserve(static_cast<size_t>(rgb.rows * rgb.cols / 2));

    for (int v = 0; v < depth.rows; ++v) {
      const auto *depth_row = depth.ptr<float>(v);
      const auto *mask_row = mask.ptr<uint8_t>(v);
      const auto *rgb_row = rgb.ptr<cv::Vec3b>(v);

      for (int u = 0; u < depth.cols; ++u) {
        const float z = depth_row[u];
        if (!std::isfinite(z) || z <= static_cast<float>(depth_min_) ||
            z >= static_cast<float>(depth_max_)) {
          continue;
        }

        const uint8_t mask_value = mask_row[u];
        if (mask_value == 255) {
          continue;
        }

        pcl::PointXYZRGBL pt;
        pt.z = z;
        pt.x = static_cast<float>((static_cast<double>(u) - cx_) * z / fx_);
        pt.y = static_cast<float>((static_cast<double>(v) - cy_) * z / fy_);

        const cv::Vec3b &bgr = rgb_row[u];
        pt.b = bgr[0];
        pt.g = bgr[1];
        pt.r = bgr[2];
        pt.label = static_cast<uint32_t>(mask_value);

        local_cloud->points.push_back(pt);
      }
    }

    if (local_cloud->points.empty()) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Local cloud is empty after filtering");
      return;
    }

    local_cloud->width = static_cast<uint32_t>(local_cloud->points.size());
    local_cloud->height = 1;
    local_cloud->is_dense = false;

    // PoseStamped 按 T_wc 使用，将局部点云从相机系变换到世界系。
    Eigen::Quaternionf q(static_cast<float>(pose_msg->pose.orientation.w),
                         static_cast<float>(pose_msg->pose.orientation.x),
                         static_cast<float>(pose_msg->pose.orientation.y),
                         static_cast<float>(pose_msg->pose.orientation.z));
    if (q.norm() < 1e-6f) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Invalid pose quaternion, skip frame");
      return;
    }
    q.normalize();

    Eigen::Affine3f t_wc = Eigen::Affine3f::Identity();
    t_wc.linear() = q.toRotationMatrix();
    t_wc.translation() = Eigen::Vector3f(static_cast<float>(pose_msg->pose.position.x),
                                         static_cast<float>(pose_msg->pose.position.y),
                                         static_cast<float>(pose_msg->pose.position.z));

    auto world_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    pcl::transformPointCloud(*local_cloud, *world_cloud, t_wc.matrix());

    *global_cloud_ += *world_cloud;

    pcl::VoxelGrid<pcl::PointXYZRGBL> voxel;
    voxel.setInputCloud(global_cloud_);
    voxel.setLeafSize(static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_),
                      static_cast<float>(voxel_leaf_size_));

    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    voxel.filter(*filtered);
    global_cloud_.swap(filtered);

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*global_cloud_, out_msg);
    out_msg.header.stamp = pose_msg->header.stamp;
    out_msg.header.frame_id = "world";
    pub_cloud_->publish(out_msg);
  }

  double fx_{};
  double fy_{};
  double cx_{};
  double cy_{};
  double voxel_leaf_size_{};
  double depth_min_{};
  double depth_max_{};

  message_filters::Subscriber<ImageMsg> rgb_sub_;
  message_filters::Subscriber<ImageMsg> depth_sub_;
  message_filters::Subscriber<ImageMsg> mask_sub_;
  message_filters::Subscriber<PoseMsg> pose_sub_;
  std::shared_ptr<Synchronizer> sync_;

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr global_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SemanticPointCloudBuilder>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
