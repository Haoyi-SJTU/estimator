// C++
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
// 多线程
#include <thread>
#include <mutex>
// ROS 独立回调队列
#include "ros/ros.h"
#include <ros/callback_queue.h>
// #include <ros/callback_queue_interface.h>
// 消息同步器
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud.h>
// tag
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h"
// 类声明
#include "estimator.hpp"
// 优化器
#include "gurobi_c++.h"

using namespace message_filters;
using namespace std;
using namespace Eigen;

// 主程序里需要做传感器时序判定（把图像的时序判定和重启拿到主程序里做）

void estimator::acc_callback(const geometry_msgs::Vector3Stamped::ConstPtr &msg) //
{
  if (!imu_init)
  {
    imu_init = 1;
    imu_t0 = (long int)msg->header.stamp.sec;
    acc_0.first = (float)(msg->header.stamp.sec - imu_t0) + (float)msg->header.stamp.nsec / 1000000000;
    acc_0.second << msg->vector.x, msg->vector.y, msg->vector.z;
    gyr_0.first = acc_0.first; // gyc跟随acc更新,用acc的时间戳
    gyr_0.second = gyr.second;
  }
  else
  {
    acc_now.first = (float)(msg->header.stamp.sec - imu_t0) + (float)msg->header.stamp.nsec / 1000000000;
    acc_now.second << msg->vector.x, msg->vector.y, msg->vector.z;
    gyr_now.first = acc_now.first; // gyc跟随acc更新,用acc的时间戳
    gyr_now.second = gyr.second;   // 陀螺仪数据跟随加速度计数据更新

    pre_integrate(); // 预积分

    acc_0.first = acc_now.first; // 更新acc
    acc_0.second = acc_now.second;
    gyr_0.first = gyr_now.first; // 更新gyr
    gyr_0.second = gyr_now.second;
  }
}

void estimator::gyr_callback(const geometry_msgs::QuaternionStamped::ConstPtr &msg) //
{
  if (!imu_init) // imu和gyc谁先到就把谁的时间戳设为0时刻
  {
    imu_init = 1;
    imu_t0 = (long int)msg->header.stamp.sec;
    gyr.first = (float)(msg->header.stamp.sec - imu_t0) + (float)msg->header.stamp.nsec / 1000000000; // 似乎不需要时间戳
    gyr.second.coeffs() << msg->quaternion.w, msg->quaternion.x, msg->quaternion.y, msg->quaternion.z;
  }
  else
  {
    gyr.first = (float)(msg->header.stamp.sec - imu_t0) + (float)msg->header.stamp.nsec / 1000000000; // 似乎不需要时间戳
    gyr.second.coeffs() << msg->quaternion.w, msg->quaternion.x, msg->quaternion.y, msg->quaternion.z;
  }
}

void estimator::feature_callback_cam1(const sensor_msgs::PointCloudConstPtr &feature_msg) //
{
  // 封装单帧图像 得到image图像消息的全部内容
  map<int, Eigen::Matrix<double, 7, 1>> image; // 存放最新图像消息特征点的 编号int、特征点信息Matrix
  int feature_id;
  double x, y, z, p_u, p_v, velocity_x, velocity_y;
  Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
  if (feature_msg->points.size() <= 3)
    return;
  for (unsigned int i = 0; i < feature_msg->points.size(); i++)
  {
    x = feature_msg->points[i].x;
    y = feature_msg->points[i].y;
    z = feature_msg->points[i].z;

    feature_id = feature_msg->channels[0].values[i];
    p_u = feature_msg->channels[1].values[i];
    p_v = feature_msg->channels[2].values[i];
    velocity_x = feature_msg->channels[3].values[i];
    velocity_y = feature_msg->channels[4].values[i];
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y; // 单个特征点的坐标、速度存入xyz_uv_velocity
    // image存放一张图的所有特征点信息
    image.emplace(feature_id, xyz_uv_velocity); // 一个feature_id对应一个xyz_uv_velocity
  }

  // 存入滑动窗的空间
  // 每存进去一个滑动窗、就把IMU数据也存进去，并把IMU数据清零
  if (img_queue.size() < WINDOW_SIZE) // 滑动窗不满
  {
    WINDOW_FULL_FLAG = 0;
    if (!add_keyframe(image)) // 图像关键帧进窗img_queue,并获取滑动窗内所有特征点编号的交集
      return;
    // 读预积分结果、清空预积分
    pthread_mutex_lock(&mutex);   // 加锁 取出预积分结果
    Ps_queue.push_back(Ps_now);   // 位置预积分进窗
    Vs_queue.push_back(Vs_now);   // 速度预积分进窗
    Rs_queue.push_back(Rs_now);   // 转角预积分进窗
    pthread_mutex_unlock(&mutex); // 解锁 结束取预积分结果
    Ps_now.setZero();             // 新一轮预积分，位移置零，速度不变。但是VINS里直接把速度置零了
    Rs_now.setZero();
  }
  else // 滑动窗已满
  {
    WINDOW_FULL_FLAG = 1;
    if (!add_keyframe(image)) // 如果符合关键帧条件，则存入滑动窗，否则退出
      return;
    // 读预积分结果、清空预积分
    pthread_mutex_lock(&mutex);   // 加锁 取出预积分结果
    Ps_queue.push_back(Ps_now);   // 位置预积分进窗
    Vs_queue.push_back(Vs_now);   // 速度预积分进窗
    Rs_queue.push_back(Rs_now);   // 转角预积分进窗
    pthread_mutex_unlock(&mutex); // 解锁 结束取预积分结果
    Ps_now.setZero();             // 新一轮预积分，位移置零，速度不变。但是VINS里直接把速度置零了
    Rs_now.setZero();
    // 滑动窗已满 需要清理掉最早的数据 对应VINS slidWindow()
    img_queue.pop_front();
    Ps_queue.pop_front();
    Vs_queue.pop_front();
    Rs_queue.pop_front();
  }

  if (WINDOW_FULL_FLAG)
  {
    if (ESTIMATOR_FLAG) // 已完成初始化 进入优化阶段
    {
      // ROS_INFO("estimator: optimization stage");
      // 这里放优化器
      calculate_reprojection_error();
    }
    else // 初始化阶段
    {
      ROS_INFO("estimator: initialization stage");
      // 这里进行初始化
      ESTIMATOR_FLAG = 1; // 完成初始化
    }
  }
}

// 计算重投影误差，加入优化器目标函数
bool estimator::calculate_reprojection_error()
{
  // std::cout << "Active_feature_id.size() = " << Active_feature_id.size() << std::endl;
  if (Active_feature_id.size() < MIN_Active_Feature) // 若小于4则不进行重投影误差计算
    return 0;
  for (int i = 0; i < WINDOW_SIZE - 1; i++)
  {
    // 找出所有活跃点的三维坐标，变换到下一帧的坐标系上
    for (const auto &id : Active_feature_id)
    {
      if (img_queue[i].find(id) != img_queue[i].end() && img_queue[i + 1].find(id) != img_queue[i + 1].end()) // 对于前帧的每一个活跃特征点
      {
        // 前后两帧归一化后的特征点坐标之差值
        Eigen::Vector3d delta_feature = img_queue[i + 1][id].head<3>() / img_queue[i + 1][id].head<3>().norm() -
                                        img_queue[i][id].head<3>() / img_queue[i][id].head<3>().norm();
        obj += robust_kernel(delta_feature, i); // 变换后的特征点坐标与后帧对应特征点之间的重投影误差
      }
    }
  }
  return 1;
}

// 鲁棒核函数 用于加权cam残差
GRBQuadExpr estimator::robust_kernel(Eigen::Vector3d delta_feature, int i)
{
  if (delta_feature.norm() > 10) // 如果误差过大则返回一范数
    return 0.5 * (delta_feature[0] * scale[i] - Ps_queue[i][0]) * (delta_feature[0] * scale[i] - Ps_queue[i][0]) +
           0.5 * (delta_feature[1] * scale[i] - Ps_queue[i][1]) * (delta_feature[1] * scale[i] - Ps_queue[i][1]) +
           0.5 * (delta_feature[2] * scale[i] - Ps_queue[i][2]) * (delta_feature[2] * scale[i] - Ps_queue[i][2]);
  else // 重投影误差的二范数
    return (delta_feature[0] * scale[i] - Ps_queue[i][0]) * (delta_feature[0] * scale[i] - Ps_queue[i][0]) +
           (delta_feature[1] * scale[i] - Ps_queue[i][1]) * (delta_feature[1] * scale[i] - Ps_queue[i][1]) +
           (delta_feature[2] * scale[i] - Ps_queue[i][2]) * (delta_feature[2] * scale[i] - Ps_queue[i][2]);
}

// 获取滑动窗内所有特征点编号的交集，作为在整个滑动窗都活跃的特征点
bool estimator::find_Active_feature_id()
{
  if (!WINDOW_FULL_FLAG) // 若滑动窗不满
    return 0;
  std::unordered_map<int, int> intersection;
  // 统计每个整数的出现次数
  for (const auto &set : img_queue)
  {
    for (const auto &pair : set)
    {
      int num = pair.first;
      if (intersection.find(num) != intersection.end())
        intersection[num]++;
      else
        intersection[num] = 1;
    }
  }
  // 找出出现次数等于队列长度的整数，将其添加到结果集合中
  std::vector<int> result;
  for (const auto &pair : intersection)
  {
    if (pair.second == WINDOW_SIZE)
      result.push_back(pair.first);
  }
  // std::cout << "Active_feature_id.size() = " << Active_feature_id.size() << std::endl;
  return 1;
}

// 测试用 发布点云消息
void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud, ros::NodeHandle &nh, ros::Publisher &pub)
{
  sensor_msgs::PointCloud2 cloudMsg;
  pcl::toROSMsg(*pointCloud, cloudMsg);
  cloudMsg.header.stamp = ros::Time::now();
  cloudMsg.header.frame_id = "world"; // 设置消息的坐标系
  pub.publish(cloudMsg);
}

// 判断特征点的图像坐标是否很靠近已知深度值的像素点
bool estimator::filterImage(const std::map<int, Eigen::Matrix<double, 7, 1>> &image,
                            double lower_bound_1, double upper_bound_1,
                            double lower_bound_2, double upper_bound_2, std::vector<int> &result)
{
  // 筛选出符合条件的编号
  // 7维向量的内容: x, y, z, p_u, p_v, velocity_x, velocity_y
  std::for_each(image.begin(), image.end(), [&](const std::pair<int, Eigen::Matrix<double, 7, 1>> &pair) -> void
                {
                  // 存在问题：两张图片的数据已经合并在一起了，cam2的数据也会进来，需要再判断一下xyz坐标，筛选出cam1的点
                  const Eigen::Matrix<double, 7, 1> &vec = pair.second;
                  bool condition_1 = (vec[3] > lower_bound_1 && vec[3] < upper_bound_1) ? 1 : 0; // 检查第四个元素是否在区间内
                  bool condition_2 = (vec[4] > lower_bound_2 && vec[4] < upper_bound_2) ? 1 : 0; // 检查第五个元素是否在区间内
                  if (condition_1 && condition_2)
                    result.push_back(pair.first); });
  if (!result.empty())
    return 1;
  else
  {
    ROS_WARN("Cannot find features near tag");
    return 0;
  }
}

// tag回调函数调用，取出tag附近的2d特征点，形成初始点云
bool estimator::pointcloud_initial(const std::map<int, Eigen::Matrix<double, 7, 1>> &image,
                                   double lower_bound_1, double upper_bound_1,
                                   double lower_bound_2, double upper_bound_2)
{
  // tag回调函数调用，取出tag附近的2d特征点，形成初始点云
  std::vector<int> pointIds;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (filterImage(image, lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2, pointIds))
  {
#pragma omp parallel for           // 并行处理区：for循环
    for (const int &id : pointIds) // 查找编号为 pointIds[i] 的点坐标
    {
      auto it = image.find(id);
      if (it != image.end())
      {
        Eigen::Matrix<double, 7, 1> point = it->second; // 将点坐标加入到 pointCloud 中
#pragma omp critical                                    // 临界区:区域内的代码同一时间只能被一个线程执行
        {
          pointCloud->push_back(pcl::PointXYZ(point(0) / 10, point(1) / 10, point(2) / 10));
        }
      }
    }
    Eigen::Matrix4d T_cam2world = T_now.inverse();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloudB(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*pointCloud, *transformedCloudB, T_cam2world);
    *pointCloud_world += *transformedCloudB; // 世界系（tag系）下的点云
    // *pointCloud_world += *pointCloud; // cam1系下的点云
    publishPointCloud(pointCloud, nh_tag, result_pub);
    // std::cout << "点云的点数是 pointCloud " << pointCloud->size() << std::endl;
    for (const auto &point : pointCloud->points)
      // std::cout << "point(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
      return 1;
  }
  else
    return 0;
}

void estimator::apriltag_callback_cam1(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg)
{
  if (msg->detections.empty())
    return;
  const std_msgs::Header &Header = msg->header;
  const std::vector<apriltag_ros::AprilTagDetection> detections = msg->detections; // 接收 AprilTagDetection[] 类型的 detections
  int num_detections = detections.size();                                          // 获取 detections 数组的大小                                                       // 存储 id 和 pose
  geometry_msgs::PoseWithCovarianceStamped pose_temp;
  Eigen::Quaterniond q_cam_tag; // tag在cam1系下的四元数
  Eigen::Vector3d t_cam_tag;    // tag在cam1系下的平移向量

  for (int i = 0; i < num_detections; i++) // 处理 detections 数据
  {
    pose_temp = detections[i].pose;
    t_cam_tag << pose_temp.pose.pose.position.x, pose_temp.pose.pose.position.y, pose_temp.pose.pose.position.z;
    q_cam_tag.coeffs() << pose_temp.pose.pose.orientation.w, pose_temp.pose.pose.orientation.x, pose_temp.pose.pose.orientation.y, pose_temp.pose.pose.orientation.z;
  }

  // 计算 T_cam_tag
  Eigen::Matrix4d T_cam_tag = Eigen::Matrix4d::Identity();
  T_cam_tag.block<3, 1>(0, 3) = t_cam_tag;
  T_cam_tag.block<3, 3>(0, 0) = q_cam_tag.toRotationMatrix();
  // 计算 T_tag_imu
  T_now = T_imu_cam.inverse() * T_cam_tag.inverse(); //  imu系在tag系下的齐次变换矩阵
  P_now = T_now.block<3, 1>(0, 3);                   //  imu系在tag系下的平移向量
  R_now = T_now.block<3, 3>(0, 0).transpose();       //  imu系在tag系下的旋转矩阵
  Q_now = Quaterniond(R_now);                        //  imu系在tag系下的四元数
}

void estimator::tag_center_callback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
  tag_center_u = msg->point.x;
  tag_center_v = msg->point.y;
  int tag_id = msg->point.z;
  float plane_range = 250;
  if (tag_id == 0) // home位放置0号tag
  {
    // 取出tag附近的2d特征点，形成初始点云
    if (!pointcloud_initial(img_queue.back(), tag_center_u - plane_range, tag_center_u + plane_range, tag_center_v - plane_range, tag_center_v + plane_range))
      ROS_ERROR("fail to add pointcloud from Apriltag -_- ");
  }
}

void estimator::laser_callback(const geometry_msgs::PointStamped::ConstPtr &msg) // laser 深度差值 回调
{
  //  主节点做一个消息回调函数，把VI优化的结果结合这个delta_depth再做优化
  //  需要做实验确定一下怎么利用两个深度数据计算H矩阵
  //  双传感器装好后，做实验在尺子上推一下：
  //  1 得到能够产生稳定H矩阵的最小平移距离
  //  2 两个相机产生两个H矩阵和两个深度差，什么时候只用一个，什么时候两个都用
  float laser_time = msg->header.stamp.toSec();
  float delta_depth_1 = msg->point.x;
  float delta_depth_2 = msg->point.y;
  if (delta_depth_1 > MIN_delta_depth_1)
  {
    if (delta_depth_2 > MIN_delta_depth_2) // 双深度预积分均达到阈值
    {
      ;
    }
    else // 仅delta_depth_1达到阈值
    {
      ;
    }
  }
  else ////仅delta_depth_2达到阈值
  {
    ;
  }
}

// 根据传入的位置、转角矩阵，更新当前末端位姿
bool estimator::refresh(Eigen::Vector3d Ps, Eigen::Matrix3d Rs)
{
  return true;
}

// IMU预积分 每次收到最新imu数据后运行 每运行一次就把最新IMU数据累加到PS、VS里
void estimator::pre_integrate()
{
  dt = acc_now.first - acc_0.first; // 两次加速度数据之间的时间差

  // 似乎不需要陀螺仪数据，因为free_acc就是世界坐标系下得到的
  // Eigen::Quaterniond gyr_now_quatern(Vector4d(gyr_now.second));
  // Eigen::Quaterniond gyr_0_quatern(Vector4d(gyr_0.second));
  // Rs[img_count] = gyr_now_quatern.toRotationMatrix();// 需要去掉gyr_0的转角!!!

  // gyr_0与gyr_now之间的转角差，转换为旋转矩阵存入Rs_now，作为预积分结果存入滑动窗
  Eigen::Quaterniond q_now = gyr_now.second * gyr_0.second.inverse();
  Rs_now = q_now.toRotationMatrix();

  // cout<<Rs[img_count]<<endl<<endl<<endl;
  // Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[img_count]) - g;
  // Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 均值滤波：上一次的加速度和当前值分别去偏差去重力后求平均

  // 加锁 更新预积分结果
  pthread_mutex_lock(&mutex);
  Ps_now += Vs_now * dt + 0.5 * dt * dt * acc_now.second;
  Vs_now += dt * acc_now.second;
  pthread_mutex_unlock(&mutex);
}

// 图像关键帧进窗img_queue,并获取滑动窗内所有特征点编号的交集
bool estimator::add_keyframe(std::map<int, Eigen::Matrix<double, 7, 1>> &image)
{
  bool imu_flag = ((Ps_now[0] > MIN_Ps_for_1_image) || (Ps_now[1] > MIN_Ps_for_1_image) || (Ps_now[2] > MIN_Ps_for_1_image) ? 1 : 0); // 如果IMU的任意一轴位置积分值超过0.05m
  // bool gyr_flag = //如果IMU的任意一轴转角积分值超过
  // bool laser_flag =
  if (imu_flag || !ESTIMATOR_FLAG) // 如果符合关键帧条件之一:1 IMU积分条件 2 激光测距条件 3 还没有进行初始化
  {
    img_queue.push_back(image);
    if (!find_Active_feature_id()) // 获取滑动窗内所有特征点编号的交集
      ROS_WARN("haven`t find Active feature id! Ignore it in Initial mode");
    return 1;
  }
  else
    return 0;
}

// 添加优化变量
void estimator::add_Variables()
{
  // 位姿、速度和四元数
  position.resize(3 * WINDOW_SIZE);
  velocity.resize(3 * WINDOW_SIZE);
  quaternion.resize(4 * WINDOW_SIZE);
  scale.resize(WINDOW_SIZE);
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    // 添加位姿变量
    position[3 * i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    position[3 * i + 1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    position[3 * i + 2] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    // 添加速度变量
    velocity[3 * i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    velocity[3 * i + 1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    velocity[3 * i + 2] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    // 添加四元数变量
    quaternion[4 * i] = model.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
    quaternion[4 * i + 1] = model.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
    quaternion[4 * i + 2] = model.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
    quaternion[4 * i + 3] = model.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
    // 添加尺度因子变量
    scale[i] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
  }
}

// 添加约束
void estimator::add_Constraints()
{
  // 模型约束
  for (int i = 0; i < WINDOW_SIZE - 1; i++)
  {
    int Q_id = 4 * i; // 四元数下标
    int P_id = 3 * i;
    // 位置约束
    model.addConstr(position[P_id] <= Max_Ps); // 约束一个窗内的最大位移 <= 0.05m
    model.addConstr(position[P_id + 1] <= Max_Ps);
    model.addConstr(position[P_id + 2] <= Max_Ps);
    model.addConstr(position[P_id + 3] - position[P_id] <= Max_Delta_Ps); // 约束相邻两个窗的最大位移差值 <= 0.01m
    model.addConstr(position[P_id + 4] - position[P_id + 1] <= Max_Delta_Ps);
    model.addConstr(position[P_id + 5] - position[P_id + 2] <= Max_Delta_Ps);
    // // 速度约束
    model.addConstr(velocity[P_id] <= Max_Vs); // 约束每帧的最大速度 <= 0.05m
    model.addConstr(velocity[P_id + 1] <= Max_Vs);
    model.addConstr(velocity[P_id + 2] <= Max_Vs);
    model.addConstr(velocity[P_id + 3] - velocity[P_id] <= Max_Delta_Vs); // 约束相邻两帧的最大速度差值 <= 0.01m
    model.addConstr(velocity[P_id + 4] - velocity[P_id + 1] <= Max_Delta_Vs);
    model.addConstr(velocity[P_id + 5] - velocity[P_id + 2] <= Max_Delta_Vs);
    // 转角四元数约束
    // model.addConstr(quaternion[Q_id] * quaternion[Q_id] + quaternion[Q_id + 1] * quaternion[Q_id + 1] + quaternion[Q_id + 2] * quaternion[Q_id + 2] + quaternion[Q_id+ 3] * quaternion[Q_id+ 3] == 1);
    // model.addConstr(quaternion_[4*i] == quaternion_[4*i + 3] * control_input_[6 * i] - quaternion_[q_idx + 2] * control_input_[6 * i + 1] + quaternion_[q_idx + 1] * control_input_[6 * i + 2]);
    // model.addConstr(quaternion_[4*i+1] == quaternion_[4*i + 2] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 1] - quaternion_[q_idx] * control_input_[6 * i + 2]);
    // model.addConstr(quaternion_[4*i+2] == -quaternion_[4*i + 1] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 2] + quaternion_[q_idx] * control_input_[6 * i + 1]);
    // model.addConstr(quaternion_[4*i+3] == -quaternion_[4*i] * control_input_[6 * i] - quaternion_[q_idx + 1] * control_input_[6 * i + 1] - quaternion_[q_idx + 2] * control_input_[6 * i + 2]);                                                                           control_input_[6 * (i - 1) + 5]);
  }
  model.addConstr(position[3 * (WINDOW_SIZE - 1)] <= Max_Ps); // 约束一个窗内的最大位移 <= 0.05m
  model.addConstr(position[3 * (WINDOW_SIZE - 1) + 1] <= Max_Ps);
  model.addConstr(position[3 * (WINDOW_SIZE - 1) + 2] <= Max_Ps);
  model.addConstr(velocity[3 * (WINDOW_SIZE - 1)] <= Max_Vs); // 约束每帧的最大速度 <= 0.05m
  model.addConstr(velocity[3 * (WINDOW_SIZE - 1) + 1] <= Max_Vs);
  model.addConstr(velocity[3 * (WINDOW_SIZE - 1) + 2] <= Max_Vs);
}

estimator::estimator() : env(), model(env)
{
  mutex = PTHREAD_MUTEX_INITIALIZER; // 多线程互斥锁 初始化
  imu_init = 0;                      // imu初始化并更新时间戳起点
  WINDOW_FULL_FLAG = 0;              // 滑动窗不满
  ESTIMATOR_FLAG = 0;                // 0:初始化阶段; 1:优化阶段
  imu_t0 = 0;
  dt = 0;
  img_count = 0;                                              // 图像计数器，在填满滑动窗时使用
  pointCloud_world.reset(new pcl::PointCloud<pcl::PointXYZ>); // 全局点云

  // cam1在imu坐标系下的变换矩阵，外参 需要标定!!!!!!!!!!!!!
  T_imu_cam << 0, 0, 1, 18.1,
      -1, 0, 0, -2.4,
      0, -1, 0, 2.9,
      0, 0, 0, 1;

  P_now << 0, 0, 0;                   // 根据tag得到的当前位置（相对于home）
  Q_now.coeffs() << 1, 0, 0, 0;       // 根据tag得到的当前转角四元数（相对于home）
  R_now << 1, 0, 0, 0, 1, 0, 0, 0, 1; // 根据tag得到的当前转角矩阵（相对于home）
  T_now.setIdentity();

  // 优化器相关
  add_Variables();   // 添加优化变量
  add_Constraints(); // 添加约束

  // 对于不同频率的消息设置独立回调队列,独立的ROS句柄
  ros::CallbackQueue queue_img, queue_tag;
  nh_img.setCallbackQueue(&queue_img);
  nh_tag.setCallbackQueue(&queue_tag);

  ros::Subscriber acc_listener = nh_imu.subscribe("/filter/free_acceleration", 1, &estimator::acc_callback, this); // IMU回调
  ros::Subscriber gyr_listener = nh_imu.subscribe("/filter/quaternion", 1, &estimator::gyr_callback, this);        // gyr回调

  ros::Subscriber img_listener_1 = nh_img.subscribe("pointcloud_talker", 1, &estimator::feature_callback_cam1, this); // 相机 图像特征点 回调

  ros::Subscriber sub = nh_tag.subscribe("/laser_distance_talker", 5, &estimator::laser_callback, this);                     // laser 深度差值 回调
  ros::Subscriber tag_listener_1 = nh_tag.subscribe("/tag_detections", 1, &estimator::apriltag_callback_cam1, this);         // 相机1 图像apriltag坐标 回调
  ros::Subscriber tag_center_listener_1 = nh_tag.subscribe("/tag_img_coordinate", 1, &estimator::tag_center_callback, this); // 相机1 图像apriltag坐标 回调

  result_pub = nh_tag.advertise<sensor_msgs::PointCloud2>("transformed_cloud", 1);

  // 每个队列内再单独设置多线程
  std::thread spinner_thread_img([&queue_img]()
                                 {ros::MultiThreadedSpinner spinner_img; spinner_img.spin(&queue_img); });
  std::thread spinner_thread_tag([&queue_tag]()
                                 {ros::MultiThreadedSpinner spinner_tag; spinner_tag.spin(&queue_tag); });

  ROS_INFO("estimator initialization finished");
  ros::spin();
  spinner_thread_img.join();
  spinner_thread_tag.join();
}

estimator::~estimator(void)
{
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "estimator");
  estimator estimator_handle; // argc, argv
  return 1;
}
