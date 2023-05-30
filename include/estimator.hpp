#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <ros/ros.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <unistd.h>
// #include <image_transport/image_transport.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <sensor_msgs/PointCloud.h>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <deque>
// #include "parameter.h"

const unsigned int WINDOW_SIZE = 10; // 滑动窗长度
const float MAX_delta_depth_1 = 200;
const float MAX_delta_depth_2 = 200;

class estimator
{
private:
	pthread_mutex_t mutex;
	ros::Timer result_pub_timer;				// 定时器
	ros::Publisher result_pub;					// 结果发布
	unsigned int img_count;						// 最新图像在滑动窗里的编号
	bool imu_init;								// 首次记录IMU
	bool WINDOW_FULL_FLAG;						// 滑动窗是否满了
	bool ESTIMATOR_FLAG;						// 0:初始化阶段; 1:优化阶段
	long int imu_t0;							// IMU时间戳初始化
	double dt;									// acc时间间隔
	std::pair<double, Eigen::Vector3d> acc_0;	// IMU 上一次 时间戳+加速度
	std::pair<double, Eigen::Vector4d> gyr_0;	// gyr 上一帧图像 时间戳+转角
	std::pair<double, Eigen::Vector3d> acc_now; // IMU 这一次 时间戳+加速度
	std::pair<double, Eigen::Vector4d> gyr_now; // gyr 取当前加速度的同时保存下来当前的 转角
	std::pair<double, Eigen::Vector4d> gyr;		// 时刻更新的 时间戳+转角
	float tag_center_u, tag_center_v;			// tag中心点的图像坐标

	// std::queue<sensor_msgs::PointCloudConstPtr> feature_msg_buf; //图像特征点消息队列
	Eigen::Vector3d Ps_now; // IMU预积分 当前三方向位置
	Eigen::Vector3d Vs_now; // IMU预积分 当前三方向速度
	Eigen::Matrix3d Rs_now; // IMU预积分 当前三方向转角

	Eigen::Vector3d P_home;	   // 记录初始点的位置向量：tag正上方0.3m处
	Eigen::Quaterniond Q_home; // 记录初始点的转角四元数：无旋转
	Eigen::Vector3d P_tag;	   // 根据tag得到的当前位置（相对于home）
	Eigen::Quaterniond Q_tag;  // 根据tag得到的当前转角四元数（相对于home）
	Eigen::Matrix3d R_tag;	   // 根据tag得到的当前转角矩阵（相对于home）

	// std::deque<map<int, vector<Eigen::Matrix<double, 7, 1> > > > img_queue; // 滑动窗 图像特征点
	// std::deque<map<int, vector<Eigen::Matrix<double, 7, 1> > > > img_queue;
	std::deque<std::map<int, std::vector<Eigen::Matrix<double, 7, 1>>>> img_queue;
	std::deque<Eigen::Vector3d> Ps_queue; // 滑动窗 三方向位置
	std::deque<Eigen::Vector3d> Vs_queue; // 滑动窗 三方向速度
	std::deque<Eigen::Vector3d> Rs_queue; // 滑动窗 三方向转角
	std::deque<Eigen::Vector3d> Bas_queue;
	std::deque<Eigen::Vector3d> Bgs_queue;
	// Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];	// 滑动窗 三方向位置
	// Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];	// 滑动窗 三方向速度
	// Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];	// 滑动窗 三方向转角
	// Eigen::Vector3d Bas[(WINDOW_SIZE + 1)]; //
	// Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)]; //

	void pre_integrate();																						  // 预积分
	bool refresh(Eigen::Vector3d Ps = Eigen::Vector3d::Zero(), Eigen::Matrix3d Rs = Eigen::Matrix3d::Identity()); // 重置估计器
	void acc_callback(const geometry_msgs::Vector3Stamped::ConstPtr &msg);
	void gyr_callback(const geometry_msgs::QuaternionStamped::ConstPtr &msg);
	void feature_callback_cam1(const sensor_msgs::PointCloudConstPtr &feature_msg); // 相机1  图像特征点 回调

	void apriltag_callback_cam1(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg); // 相机1 图像apriltag坐标 回调
	void tag_center_callback(const geometry_msgs::PointStamped::ConstPtr &msg);				// 接收tag中心的图像坐标
	bool add_keyframe(std::map<int, std::vector<Eigen::Matrix<double, 7, 1>>> &);			// 向滑动窗添加关键帧

	void laser_callback(const geometry_msgs::PointStamped::ConstPtr &msg); // laser 深度差值 回调

public:
	estimator();
	~estimator();
};
