#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "adaptive_clustering/ClusterArray.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

// **TF相关头文件**
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PointStamped.h>

ros::Publisher cluster_array_pub_;    //发布聚类后的点云数组
ros::Publisher cloud_filtered_pub_;     //发布滤波后的点云
ros::Publisher pose_array_pub_;   //发布聚类中心点位姿
ros::Publisher marker_array_pub_;    // 发布可视化标记
ros::Publisher bbox_corners_pub_;    // **新增：发布边界框角点点云**

bool print_fps_;     // 是否打印帧率
int leaf_;    // 降采样因子
float z_axis_min_;   // 最小高度，去除地面
float z_axis_max_;   // 最大高度，去除天花板
int cluster_size_min_;   // 最小聚类大小
int cluster_size_max_;   // 最大聚类大小

const int region_max_ = 10; // 聚类区域大小
int regions_[100];  // 定义嵌套环形区域大小

int frames; clock_t start_time; bool reset = true;//fps

// 旋转矩阵
Eigen::Matrix3f rotation_matrix_;

// **TF监听器**
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

// **坐标系参数**
std::string target_frame_;  // 目标坐标系
std::string source_frame_;  // 源坐标系

// 初始化旋转矩阵
void initializeRotationMatrix() {
    rotation_matrix_ << 0.939f, 0.0f,   0.342f,
                        0.0f,   1.0f,   0.0f,
                       -0.342f, 0.0f,   0.939f;
}

// **将点从base_link转换到map坐标系**
geometry_msgs::Point transformPoint(const geometry_msgs::Point& point_in, 
                                   const ros::Time& timestamp) {
    geometry_msgs::Point point_out = point_in;
    
    try {
        // 创建PointStamped消息
        geometry_msgs::PointStamped point_stamped_in, point_stamped_out;
        point_stamped_in.header.frame_id = source_frame_;
        point_stamped_in.header.stamp = timestamp;
        point_stamped_in.point = point_in;
        
        // 执行坐标变换
        point_stamped_out = tf_buffer_->transform(point_stamped_in, target_frame_);
        point_out = point_stamped_out.point;
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "TF变换失败: %s", ex.what());
        // 变换失败时返回原始点
    }
    
    return point_out;
}

// **获取变换矩阵**
bool getTransform(const ros::Time& timestamp, Eigen::Matrix4f& transform_matrix) {
    try {
        geometry_msgs::TransformStamped transform_stamped = 
            tf_buffer_->lookupTransform(target_frame_, source_frame_, timestamp, ros::Duration(0.1));
        
        // 提取平移
        Eigen::Vector3f translation(
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z
        );
        
        // 提取旋转（四元数转旋转矩阵）
        Eigen::Quaternionf quaternion(
            transform_stamped.transform.rotation.w,
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z
        );
        
        // 构建4x4变换矩阵
        transform_matrix = Eigen::Matrix4f::Identity();
        transform_matrix.block<3,3>(0,0) = quaternion.toRotationMatrix();
        transform_matrix.block<3,1>(0,3) = translation;
        
        return true;
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "无法获取变换矩阵: %s", ex.what());
        return false;
    }
}

// **新增：创建边界框角点点云**
pcl::PointCloud<pcl::PointXYZI>::Ptr createBBoxCornersPointCloud(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr>>& clusters,
    const ros::Time& timestamp) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    for (size_t i = 0; i < clusters.size(); i++) {
        // 计算当前聚类的边界框
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*clusters[i], min_pt, max_pt);
        
        // 创建min corner点
        pcl::PointXYZI min_corner;
        min_corner.x = min_pt[0];
        min_corner.y = min_pt[1];
        min_corner.z = min_pt[2]-1.5;
        min_corner.intensity = 0.0f;  // intensity=0 表示min corner
        
        // 创建max corner点
        pcl::PointXYZI max_corner;
        max_corner.x = max_pt[0];
        max_corner.y = max_pt[1];
        max_corner.z = max_pt[2];
        max_corner.intensity = 1.0f;  // intensity=1 表示max corner
        
        // 将corner点变换到map坐标系
        geometry_msgs::Point min_point, max_point;
        min_point.x = min_corner.x;
        min_point.y = min_corner.y;
        min_point.z = min_corner.z;
        
        max_point.x = max_corner.x;
        max_point.y = max_corner.y;
        max_point.z = max_corner.z;
        
        // 变换到map坐标系
        geometry_msgs::Point transformed_min = transformPoint(min_point, timestamp);
        geometry_msgs::Point transformed_max = transformPoint(max_point, timestamp);
        
        // 更新变换后的坐标
        min_corner.x = transformed_min.x;
        min_corner.y = transformed_min.y;
        min_corner.z = transformed_min.z;
        
        max_corner.x = transformed_max.x;
        max_corner.y = transformed_max.y;
        max_corner.z = transformed_max.z;
        
        // 添加到点云中
        corners_cloud->points.push_back(min_corner);
        corners_cloud->points.push_back(max_corner);
    }
    
    // 设置点云属性
    corners_cloud->width = corners_cloud->points.size();
    corners_cloud->height = 1;
    corners_cloud->is_dense = true;
    
    ROS_DEBUG("创建边界框角点点云，聚类数: %lu, 角点数: %lu", 
             clusters.size(), corners_cloud->points.size());
    
    return corners_cloud;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr transformPointCloudToMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    const ros::Time& timestamp) {
    
    // **修复：使用PCL兼容的boost::shared_ptr**
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    // 获取TF变换矩阵
    Eigen::Matrix4f tf_transform_matrix;
    bool has_transform = getTransform(timestamp, tf_transform_matrix);
    
    if (has_transform) {
        // 使用PCL的变换函数将整个点云变换到map坐标系
        pcl::transformPointCloud(*input_cloud, *output_cloud, tf_transform_matrix);
        ROS_DEBUG("点云已变换到map坐标系，点数: %lu", output_cloud->size());
    } else {
        // TF变换失败时，复制原始点云
        pcl::copyPointCloud(*input_cloud, *output_cloud);
        ROS_WARN_THROTTLE(5.0, "TF变换失败，发布原始坐标系点云");
    }
    
    return output_cloud;
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) 
{
  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}
  
  // ROS格式转换为PCL格式
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
  
  // **点云旋转变换 - 在降采样和滤除之前执行**
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_rotated(new pcl::PointCloud<pcl::PointXYZI>);
  
  // 创建4x4变换矩阵
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3,3>(0,0) = rotation_matrix_;
  
  // 应用旋转变换
  pcl::transformPointCloud(*pcl_pc_in, *pcl_pc_rotated, transform);
  
  ROS_DEBUG("点云旋转完成，原始点数: %lu, 旋转后点数: %lu", 
           pcl_pc_in->size(), pcl_pc_rotated->size());
  
  // 降采样+地面&天花板滤除 - 现在使用旋转后的点云
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  for(int i = 0; i < pcl_pc_rotated->size(); ++i) {
    if(i % leaf_ == 0) {
      if(pcl_pc_rotated->points[i].z >= z_axis_min_ && pcl_pc_rotated->points[i].z <= z_axis_max_) {
        pc_indices->push_back(i);
      }
    }
  }
  
  /***将点云划分为嵌套的圆形区域***/
  boost::array<std::vector<int>, region_max_> indices_array;
  for(int i = 0; i < pc_indices->size(); i++) {
    float range = 0.0;
    for(int j = 0; j < region_max_; j++) {
      float d2 = pcl_pc_rotated->points[(*pc_indices)[i]].x * pcl_pc_rotated->points[(*pc_indices)[i]].x +
        pcl_pc_rotated->points[(*pc_indices)[i]].y * pcl_pc_rotated->points[(*pc_indices)[i]].y +
        pcl_pc_rotated->points[(*pc_indices)[i]].z * pcl_pc_rotated->points[(*pc_indices)[i]].z;
      if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
        indices_array[j].push_back((*pc_indices)[i]);
        break;
      }
      range += regions_[j];
    }
  }
  
  /*** 欧式聚类 ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr > > clusters;
  
  for(int i = 0; i < region_max_; i++) {
    tolerance += 0.1;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pcl_pc_rotated, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_rotated);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
          cluster->points.push_back(pcl_pc_rotated->points[*pit]);
        }
        cluster->width = cluster->size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
      }
    }
  }
  
  /*** PCL格式转ROS格式并发布 ***/
  // **关键修改：发布变换到map坐标系的过滤点云**
  if(cloud_filtered_pub_.getNumSubscribers() > 0) {
    // 1. 首先从旋转后的点云中提取过滤后的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pcl_pc_rotated, *pc_indices, *pcl_pc_filtered);
    
    // 2. 将过滤后的点云变换到map坐标系
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_map = 
        transformPointCloudToMap(pcl_pc_filtered, ros_pc2_in->header.stamp);
    
    // 3. 转换为ROS消息并发布
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::toROSMsg(*pcl_pc_map, ros_pc2_out);
    ros_pc2_out.header = ros_pc2_in->header;
    ros_pc2_out.header.frame_id = target_frame_;  // **重要：发布到map坐标系**
    cloud_filtered_pub_.publish(ros_pc2_out);
    
    ROS_DEBUG("发布过滤点云到map坐标系，点数: %lu", pcl_pc_map->size());
  }
  
  adaptive_clustering::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;
  
  // **获取TF变换矩阵用于边界框变换**
  Eigen::Matrix4f tf_transform_matrix;
  bool has_transform = getTransform(ros_pc2_in->header.stamp, tf_transform_matrix);
  
  // **新增：发布边界框角点点云**
  if(bbox_corners_pub_.getNumSubscribers() > 0 && clusters.size() > 0) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cloud = 
        createBBoxCornersPointCloud(clusters, ros_pc2_in->header.stamp);
    
    sensor_msgs::PointCloud2 corners_msg;
    pcl::toROSMsg(*corners_cloud, corners_msg);
    corners_msg.header = ros_pc2_in->header;
    corners_msg.header.frame_id = target_frame_;  // 发布到map坐标系
    bbox_corners_pub_.publish(corners_msg);
    
    ROS_DEBUG("发布边界框角点点云，点数: %lu", corners_cloud->size());
  }
  
  for(int i = 0; i < clusters.size(); i++) {
    // **聚类数组：可选择发布到哪个坐标系**
    if(cluster_array_pub_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 ros_pc2_out;
      
      // **选项1：将聚类也变换到map坐标系**
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_map = 
          transformPointCloudToMap(clusters[i], ros_pc2_in->header.stamp);
      pcl::toROSMsg(*cluster_map, ros_pc2_out);
      ros_pc2_out.header = ros_pc2_in->header;
      ros_pc2_out.header.frame_id = target_frame_;  // 聚类也发布到map坐标系
      
      // **选项2：保持聚类在base_link系（注释掉上面4行，取消注释下面3行）**
      // pcl::toROSMsg(*clusters[i], ros_pc2_out);
      // ros_pc2_out.header = ros_pc2_in->header;
      // ros_pc2_out.header.frame_id = source_frame_;  // 聚类保持在base_link系
      
      cluster_array.clusters.push_back(ros_pc2_out);
    }
    
    if(pose_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*clusters[i], centroid);
      
      geometry_msgs::Pose pose;
      
      if(has_transform) {
        // **变换质心到map坐标系**
        geometry_msgs::Point centroid_point;
        centroid_point.x = centroid[0];
        centroid_point.y = centroid[1];
        centroid_point.z = centroid[2];
        geometry_msgs::Point transformed_centroid = transformPoint(centroid_point, ros_pc2_in->header.stamp);
        
        pose.position = transformed_centroid;
      } else {
        // TF变换失败时使用原始坐标
        pose.position.x = centroid[0];
        pose.position.y = centroid[1];
        pose.position.z = centroid[2];
      }
      
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);
    }
    
    if(marker_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      
      visualization_msgs::Marker marker;
      marker.header = ros_pc2_in->header;
      marker.header.frame_id = target_frame_;  // **边界框发布到map坐标系**
      marker.ns = "adaptive_clustering";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.pose.position.x = 0.0;
      marker.pose.position.y = 0.0;
      marker.pose.position.z = 0.0;

      // **变换边界框的8个顶点到map坐标系**
      geometry_msgs::Point bbox_corners[8];
      
      // 定义8个顶点（在base_link坐标系中）
      bbox_corners[0].x = min[0]; bbox_corners[0].y = min[1]; bbox_corners[0].z = min[2]; // min corner
      bbox_corners[1].x = max[0]; bbox_corners[1].y = min[1]; bbox_corners[1].z = min[2];
      bbox_corners[2].x = max[0]; bbox_corners[2].y = max[1]; bbox_corners[2].z = min[2];
      bbox_corners[3].x = min[0]; bbox_corners[3].y = max[1]; bbox_corners[3].z = min[2];
      bbox_corners[4].x = min[0]; bbox_corners[4].y = min[1]; bbox_corners[4].z = max[2];
      bbox_corners[5].x = max[0]; bbox_corners[5].y = min[1]; bbox_corners[5].z = max[2];
      bbox_corners[6].x = max[0]; bbox_corners[6].y = max[1]; bbox_corners[6].z = max[2]; // max corner
      bbox_corners[7].x = min[0]; bbox_corners[7].y = max[1]; bbox_corners[7].z = max[2];
      
      // 变换所有顶点到map坐标系
      for(int k = 0; k < 8; k++) {
        if(has_transform) {
          bbox_corners[k] = transformPoint(bbox_corners[k], ros_pc2_in->header.stamp);
        }
      }
      
      // **构建边界框的12条边**
      geometry_msgs::Point p[24];
      // 底面4条边
      p[0] = bbox_corners[0]; p[1] = bbox_corners[1];  // min-x边
      p[2] = bbox_corners[1]; p[3] = bbox_corners[2];  // min-y边
      p[4] = bbox_corners[2]; p[5] = bbox_corners[3];  // max-x边
      p[6] = bbox_corners[3]; p[7] = bbox_corners[0];  // max-y边
      
      // 顶面4条边
      p[8] = bbox_corners[4]; p[9] = bbox_corners[5];
      p[10] = bbox_corners[5]; p[11] = bbox_corners[6];
      p[12] = bbox_corners[6]; p[13] = bbox_corners[7];
      p[14] = bbox_corners[7]; p[15] = bbox_corners[4];
      
      // 垂直4条边
      p[16] = bbox_corners[0]; p[17] = bbox_corners[4];
      p[18] = bbox_corners[1]; p[19] = bbox_corners[5];
      p[20] = bbox_corners[2]; p[21] = bbox_corners[6];
      p[22] = bbox_corners[3]; p[23] = bbox_corners[7];
      
      for(int j = 0; j < 24; j++) {
        marker.points.push_back(p[j]);
      }
      
      marker.scale.x = 0.02;
      marker.color.a = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.5;
      marker.lifetime = ros::Duration(0.1);
      marker_array.markers.push_back(marker);
    }
  }
  
  if(cluster_array.clusters.size()) {
    cluster_array.header = ros_pc2_in->header;
    cluster_array.header.frame_id = target_frame_;  // **修改：聚类数组发布到map坐标系**
    cluster_array_pub_.publish(cluster_array);
  }

  if(pose_array.poses.size()) {
    pose_array.header = ros_pc2_in->header;
    pose_array.header.frame_id = target_frame_;  // **质心发布到map坐标系**
    pose_array_pub_.publish(pose_array);
  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  
  if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps
}

int main(int argc, char **argv) 
{
  ros::init(argc, argv, "adaptive_clustering");
  
  // 初始化旋转矩阵
  initializeRotationMatrix();
  ROS_INFO("initializeRotationMatrix completed.");
  
  // **初始化TF监听器**
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  
  // 订阅
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/livox/point_cloud", 1, pointCloudCallback);

  // 发布
  ros::NodeHandle private_nh("~");
  cluster_array_pub_ = private_nh.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
  bbox_corners_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("/global_planning_obs_node/obs_cost", 100);  // **新增：边界框角点话题**
  
  // 参数
  std::string sensor_model;
  
  private_nh.param<std::string>("sensor_model", sensor_model, "HDL-32E"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<int>("leaf", leaf_, 1);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.3);
  private_nh.param<float>("z_axis_max", z_axis_max_, 2.0);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 20);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 200000);
  
  // **TF坐标系参数**
  private_nh.param<std::string>("target_frame", target_frame_, "map");
  private_nh.param<std::string>("source_frame", source_frame_, "base_link");
  
  ROS_INFO("TF变换设置: %s -> %s", source_frame_.c_str(), target_frame_.c_str());
  ROS_INFO("过滤点云将发布到: %s 坐标系", target_frame_.c_str());
  ROS_INFO("边界框角点将发布到话题: /adaptive_clustering/bbox_corners");
  
  // 将点云划分成以传感器为中心的嵌套圆形区域
  if(sensor_model.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  
  // **等待TF变换可用**
  ROS_INFO("等待TF变换 %s -> %s 可用...", source_frame_.c_str(), target_frame_.c_str());
  try {
    tf_buffer_->canTransform(target_frame_, source_frame_, ros::Time(0), ros::Duration(10.0));
    ROS_INFO("TF变换已准备就绪");
  } catch (tf2::TransformException& ex) {
    ROS_WARN("TF变换等待超时: %s", ex.what());
    ROS_WARN("程序将继续运行，但点云可能不会正确变换");
  }
  
  ros::spin();

  return 0;
}