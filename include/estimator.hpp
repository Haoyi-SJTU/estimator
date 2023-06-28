#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <ros/ros.h>

#include <geometry_msgs/Vector3Stamped.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h> //测试发布ROS点云消息用
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <deque>
// 点云处理
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h> //测试发布ROS点云消息用
// 优化器
#include "gurobi_c++.h"

// 获取活跃特征点
#include <unordered_map>
#include <algorithm>

const unsigned int WINDOW_SIZE = 10;   // 滑动窗长度
const float MIN_delta_depth_1 = 200;   // 深度预积分阈值
const float MIN_delta_depth_2 = 200;   // 深度预积分阈值
const float MIN_Ps_for_1_image = 0.05; // 能让图像成为关键帧的最小IMU积分阈值 0.05 m
const float Max_Ps = 0.05;			   // 一个窗内的最大位移 单位m
const float Max_Delta_Ps = 0.01;	   // 相邻两个窗的最大位移差值 单位m
const float Max_Vs = 5;				   // 每帧的最大速度 单位m/s
const float Max_Delta_Vs = 1;		   // 相邻两帧的最大速度差值 单位m/s
const float MIN_Active_Feature = 4;	   // 活跃特征点个数的最小阈值，小于4则不进行重投影误差计算

class estimator
{
private:
	pthread_mutex_t mutex;
	ros::Timer result_pub_timer;				   // 定时器
	ros::Publisher result_pub;					   // 结果发布
	ros::NodeHandle nh_imu;						   // IMU消息相应句柄
	ros::NodeHandle nh_img;						   // 图像消息响应句柄
	ros::NodeHandle nh_tag;						   // tag消息、tag中心坐标、激光点达到预积分阈值响应句柄
	unsigned int img_count;						   // 最新图像在滑动窗里的编号
	bool imu_init;								   // 首次记录IMU
	bool WINDOW_FULL_FLAG;						   // 滑动窗是否满了
	bool ESTIMATOR_FLAG;						   // 0:初始化阶段; 1:优化阶段
	long int imu_t0;							   // IMU时间戳初始化
	double dt;									   // acc时间间隔
	std::pair<double, Eigen::Vector3d> acc_0;	   // IMU 上一次 时间戳+加速度
	std::pair<double, Eigen::Quaterniond> gyr_0;   // gyr 上一帧图像 时间戳+转角
	std::pair<double, Eigen::Vector3d> acc_now;	   // IMU 这一次 时间戳+加速度
	std::pair<double, Eigen::Quaterniond> gyr_now; // gyr 取当前加速度的同时保存下来当前的 转角 (是转角不是角速度)
	std::pair<double, Eigen::Quaterniond> gyr;	   // 时刻更新的 时间戳+转角
	float tag_center_u, tag_center_v;			   // tag中心点的图像坐标

	// std::queue<sensor_msgs::PointCloudConstPtr> feature_msg_buf; //图像特征点消息队列
	Eigen::Vector3d Ps_now;								  // IMU预积分 当前三方向位置
	Eigen::Vector3d Vs_now;								  // IMU预积分 当前三方向速度
	Eigen::Matrix3d Rs_now;								  // IMU预积分 当前三方向转角
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_world; // 世界坐标系下的点云

	Eigen::Matrix4d T_imu_cam; // cam1相对于imu坐标系的变换矩阵，外参   注意此参数来自于标定！！！！！！！！！！！！！！
	Eigen::Vector3d P_now;	   // 当前位置（相对于tag）
	Eigen::Quaterniond Q_now;  // 当前转角四元数（相对于tag）
	Eigen::Matrix3d R_now;	   // 当前转角矩阵（相对于tag）
	Eigen::Matrix4d T_now;	   // 当前齐次变换矩阵（相对于tag）

	std::deque<std::map<int, Eigen::Matrix<double, 7, 1>>> img_queue; // 滑动窗 图像特征点
	std::deque<Eigen::Vector3d> Ps_queue;							  // 滑动窗 三方向位置
	std::deque<Eigen::Vector3d> Vs_queue;							  // 滑动窗 三方向速度
	std::deque<Eigen::Matrix3d> Rs_queue;							  // 滑动窗 转角四元数
	std::deque<Eigen::Vector3d> Bas_queue;
	std::deque<Eigen::Vector3d> Bgs_queue;

	std::vector<int> Active_feature_id; // 动态更新在整个滑动窗内都活跃的特征点ID

	GRBEnv env;						// 优化环境
	GRBModel model;					// 优化模型
	GRBQuadExpr obj;				// 定义目标函数
	std::vector<GRBVar> position;	// 优化变量: 位姿
	std::vector<GRBVar> velocity;	// 优化变量: 速度
	std::vector<GRBVar> quaternion; // 优化变量: 四元数
	std::vector<GRBVar> scale;		// 优化变量: 尺度因子

	void pre_integrate();																						  // 预积分
	bool refresh(Eigen::Vector3d Ps = Eigen::Vector3d::Zero(), Eigen::Matrix3d Rs = Eigen::Matrix3d::Identity()); // 重置估计器
	bool pointcloud_initial(const std::map<int, Eigen::Matrix<double, 7, 1>> &, double, double, double, double);  // 初始化3D点云

	bool filterImage(const std::map<int, Eigen::Matrix<double, 7, 1>> &, double, double, double, double, std::vector<int> &);
	bool add_keyframe(std::map<int, Eigen::Matrix<double, 7, 1>> &); // 向滑动窗添加关键帧
	bool find_Active_feature_id();									 // 获取滑动窗内所有特征点编号的交集，作为在整个滑动窗都活跃的特征点

	// 优化器相关函数
	void add_Variables();						// 添加优化变量
	void add_Constraints();						// 添加约束条件
	bool calculate_reprojection_error();		// cam残差: 将重投影误差加入目标函数
	GRBQuadExpr robust_kernel(Eigen::Vector3d, int); // 鲁棒核函数 用于加权cam残差

	void acc_callback(const geometry_msgs::Vector3Stamped::ConstPtr &msg);
	void gyr_callback(const geometry_msgs::QuaternionStamped::ConstPtr &msg);
	void feature_callback_cam1(const sensor_msgs::PointCloudConstPtr &feature_msg);			// 相机1  图像特征点 回调
	void apriltag_callback_cam1(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg); // 相机1 图像apriltag坐标 回调
	void tag_center_callback(const geometry_msgs::PointStamped::ConstPtr &msg);				// 接收tag中心的图像坐标

	void laser_callback(const geometry_msgs::PointStamped::ConstPtr &msg); // laser 深度差值 回调

public:
	estimator();
	~estimator();
};
