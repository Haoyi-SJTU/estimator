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
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h"
// 类声明
#include "estimator.hpp"

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
    gyr.second << msg->quaternion.x, msg->quaternion.y, msg->quaternion.z, msg->quaternion.w;
  }
  else
  {
    gyr.first = (float)(msg->header.stamp.sec - imu_t0) + (float)msg->header.stamp.nsec / 1000000000; // 似乎不需要时间戳
    gyr.second << msg->quaternion.x, msg->quaternion.y, msg->quaternion.z, msg->quaternion.w;
  }
}

void estimator::feature_callback_cam1(const sensor_msgs::PointCloudConstPtr &feature_msg) //
{
  // feature_msg_buf.push(feature_msg);//图像消息存储队列 用不到，因为消息接收自带队列
  // if (feature_msg_buf.size() >= WINDOW_SIZE) // 移除超过滑动窗范围的图像
  //   feature_msg_buf.pop();

  // 封装单帧图像 得到image图像消息的全部内容
  map<int, vector<Eigen::Matrix<double, 7, 1>>> image; // 存放最新图像消息特征点的 编号int、特征点信息Matrix
  int feature_id;
  double x, y, z, p_u, p_v, velocity_x, velocity_y;
  Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
  if (feature_msg->points.size() <= 3)
    return;
  for (unsigned int i = 0; i < feature_msg->points.size(); i++)
  {
    feature_id = feature_msg->channels[0].values[i];
    x = feature_msg->points[i].x;
    y = feature_msg->points[i].y;
    z = feature_msg->points[i].z;
    p_u = feature_msg->channels[0].values[i]; // 取消了特征点ID项,如果后面加入ID需要修改这里读取channels的编号
    p_v = feature_msg->channels[1].values[i];
    velocity_x = feature_msg->channels[2].values[i];
    velocity_y = feature_msg->channels[3].values[i];
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y; // 单个特征点的坐标、速度存入xyz_uv_velocity
    image[feature_id].emplace_back(xyz_uv_velocity);              // xyz_uv_velocity加入image,image存放一张图的所有特征点信息
  }

  // 存入滑动窗的空间
  // 每存进去一个滑动窗、就把IMU数据也存进去，并把IMU数据清零
  if (img_queue.size() < WINDOW_SIZE) // 滑动窗不满
  {
    WINDOW_FULL_FLAG = 0;
    if (!add_keyframe(image)) // 如果符合关键帧条件，则存入滑动窗，否则退出
      return;
    pthread_mutex_lock(&mutex); // 加锁 取出预积分结果
    Ps_queue.push_back(Ps_now); // 读预积分结果、清空预积分
    Vs_queue.push_back(Vs_now);
    pthread_mutex_unlock(&mutex); // 解锁 结束取预积分结果
    Ps_now.setZero();             // 新一轮预积分，位移置零，速度不变。但是VINS里直接把速度置零了
  }
  else // 滑动窗已满
  {
    WINDOW_FULL_FLAG = 1;
    if (!add_keyframe(image)) // 如果符合关键帧条件，则存入滑动窗，否则退出
      return;
    pthread_mutex_lock(&mutex); // 加锁 取出预积分结果
    Ps_queue.push_back(Ps_now); // 读预积分结果、清空预积分
    Vs_queue.push_back(Vs_now);
    pthread_mutex_unlock(&mutex); // 解锁 结束取预积分结果
    Ps_now.setZero();             // 新一轮预积分，位移置零，速度不变。但是VINS里直接把速度置零了
    // 滑动窗已满 需要清理掉最早的数据 对应VINS slidWindow()
    img_queue.pop_front();
    Ps_queue.pop_front();
    Vs_queue.pop_front();
  }

  if (WINDOW_FULL_FLAG)
  {
    if (ESTIMATOR_FLAG) // 已完成初始化 进入优化阶段
    {
      ROS_INFO("estimator: optimization stage");
    }
    else // 初始化阶段
    {
      ROS_INFO("estimator: initialization stage");
      // 这里进行初始化
      ESTIMATOR_FLAG = 1; // 完成初始化
    }
  }
}

void estimator::apriltag_callback_cam1(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg)
{
  const std_msgs::Header &Header = msg->header;
  const std::vector<apriltag_ros::AprilTagDetection> detections = msg->detections; // 接收 AprilTagDetection[] 类型的 detections
  int num_detections = detections.size();                                          // 获取 detections 数组的大小
  std::vector<int32_t> ids;                                                        // 存储 id 和 pose
  geometry_msgs::PoseWithCovarianceStamped pose_temp;
  Eigen::Quaterniond Q_temp;
  Eigen::Vector3d P_temp;
  for (int i = 0; i < num_detections; i++) // 处理 detections 数据
  {
    pose_temp = detections[i].pose;
    if (ids.empty())
      return;
    P_temp << pose_temp.pose.pose.position.x, pose_temp.pose.pose.position.y, pose_temp.pose.pose.position.z;
    Q_temp.coeffs() << pose_temp.pose.pose.orientation.w, pose_temp.pose.pose.orientation.x, pose_temp.pose.pose.orientation.y, pose_temp.pose.pose.orientation.z;
  }
  P_tag = P_temp - P_home;
  Q_tag = Q_home * Q_temp.inverse();
  R_tag = Q_tag.toRotationMatrix();
}

void estimator::tag_center_callback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
  tag_center_u = msg->point.x;
  tag_center_v = msg->point.y;
  int tag_id = msg->point.z;
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
  if (delta_depth_1 > MAX_delta_depth_1)
  {
    if (delta_depth_2 > MAX_delta_depth_2) // 双深度预积分均达到阈值
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

  // cout<<Rs[img_count]<<endl<<endl<<endl;
  // Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[img_count]) - g;
  // Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 均值滤波：上一次的加速度和当前值分别去偏差去重力后求平均

  // 加锁 更新预积分结果
  pthread_mutex_lock(&mutex);
  Ps_now += Vs_now * dt + 0.5 * dt * dt * acc_now.second;
  Vs_now += dt * acc_now.second;
  pthread_mutex_unlock(&mutex);
}

bool estimator::add_keyframe(std::map<int, std::vector<Eigen::Matrix<double, 7, 1>>> &image)
{
  bool imu_flag = ((Ps_now[0]>0.05)||(Ps_now[1]>0.05)||(Ps_now[2]>0.05)? 1:0);//如果IMU的任意一轴积分值超过0.05m
  // bool gyr_flag = 
  // bool laser_flag = 
  if (imu_flag || !ESTIMATOR_FLAG)//如果符合关键帧条件之一:1 IMU积分条件 2 激光测距条件 3 还没有进行初始化
  {
    img_queue.push_back(image);
    return 1;
  }
  else
    return 0;
}

estimator::estimator()
{
  mutex = PTHREAD_MUTEX_INITIALIZER; // 多线程互斥锁 初始化
  imu_init = 0;                      // imu初始化并更新时间戳起点
  WINDOW_FULL_FLAG = 0;              // 滑动窗不满
  ESTIMATOR_FLAG = 0;                // 0:初始化阶段; 1:优化阶段
  imu_t0 = 0;
  dt = 0;
  img_count = 0;                      // 图像计数器，在填满滑动窗时使用
  P_home << 0, 0, 0.3;                // 记录初始点的位置向量：tag正上方0.3m处
  Q_home.coeffs() << 1, 0, 0, 0;      // 记录初始点的转角四元数：无旋转
  P_tag << 0, 0, 0;                   // 根据tag得到的当前位置（相对于home）
  Q_tag.coeffs() << 1, 0, 0, 0;       // 根据tag得到的当前转角四元数（相对于home）
  R_tag << 1, 0, 0, 0, 1, 0, 0, 0, 1; // 根据tag得到的当前转角矩阵（相对于home）

  // 对于不同频率的消息设置独立回调队列,独立的ROS句柄
  ros::NodeHandle nh_imu;
  ros::NodeHandle nh_img;
  ros::NodeHandle nh_tag;
  ros::CallbackQueue queue_img, queue_tag;
  nh_img.setCallbackQueue(&queue_img);
  nh_tag.setCallbackQueue(&queue_tag);

  ros::Subscriber acc_listener = nh_imu.subscribe("/filter/free_acceleration", 1, &estimator::acc_callback, this); // IMU回调
  ros::Subscriber gyr_listener = nh_imu.subscribe("/filter/quaternion", 1, &estimator::gyr_callback, this);        // gyr回调

  ros::Subscriber img_listener_1 = nh_img.subscribe("pointcloud_talker", 1, &estimator::feature_callback_cam1, this); // 相机 图像特征点 回调

  ros::Subscriber sub = nh_tag.subscribe("/laser_distance_talker", 5, &estimator::laser_callback, this);                     // laser 深度差值 回调
  ros::Subscriber tag_listener_1 = nh_tag.subscribe("/tag_detections", 1, &estimator::apriltag_callback_cam1, this);         // 相机1 图像apriltag坐标 回调
  ros::Subscriber tag_center_listener_1 = nh_tag.subscribe("/tag_img_coordinate", 1, &estimator::tag_center_callback, this); // 相机1 图像apriltag坐标 回调

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
