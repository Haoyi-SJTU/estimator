#include "estimator.hpp"


// 测试用 发布点云消息
void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud, ros::NodeHandle &nh, ros::Publisher &pub)
{
  sensor_msgs::PointCloud2 cloudMsg;
  pcl::toROSMsg(*pointCloud, cloudMsg);
  cloudMsg.header.stamp = ros::Time::now();
  cloudMsg.header.frame_id = "world"; // 设置消息的坐标系
  pub.publish(cloudMsg);
}

// 打印所有优化结果
void print_result_debug(const unsigned int WINDOW_SIZE,
                       std::vector<GRBVar> &position, std::vector<GRBVar> &velocity,
                       std::vector<GRBVar> &quaternion, std::vector<GRBVar> &scale)
{
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    double p_x = position[3 * i].get(GRB_DoubleAttr_X) / 10;
    double p_y = position[3 * i + 1].get(GRB_DoubleAttr_X) / 10;
    double p_z = position[3 * i + 2].get(GRB_DoubleAttr_X) / 10;
    double v_x = velocity[3 * i].get(GRB_DoubleAttr_X) / 10;
    double v_y = velocity[3 * i + 1].get(GRB_DoubleAttr_X) / 10;
    double v_z = velocity[3 * i + 2].get(GRB_DoubleAttr_X) / 10;
    double q_w = quaternion[4 * i].get(GRB_DoubleAttr_X);
    double q_x = quaternion[4 * i + 1].get(GRB_DoubleAttr_X);
    double q_y = quaternion[4 * i + 2].get(GRB_DoubleAttr_X);
    double q_z = quaternion[4 * i + 3].get(GRB_DoubleAttr_X);
    double scale_1 = scale[2 * i].get(GRB_DoubleAttr_X);
    double scale_2 = scale[2 * i + 1].get(GRB_DoubleAttr_X);
    std::cout << "优化后 p=(" << p_x << ", " << p_y << ", " << p_z << "), v=("
              << v_x << ", " << v_y << ", " << v_z << "), q=("
              << q_w << ", " << q_x << ", " << q_y << ", " << q_z << "), scale=("
              << scale_1 << ", " << scale_2 << ")" << std::endl;
  }
}
