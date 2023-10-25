#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct Point3D
{
     float x;
     float y;
     float z;
};

int main(int argc, char **argv)
{
     ros::init(argc, argv, "ideal_trace_node");
     ros::NodeHandle nh;
     ros::Publisher pose_pub = nh.advertise<geometry_msgs::PointStamped>("ideal_trace", 1);

     std::ifstream file("/home/yuanzhi/catkin_ws/src/estimator/data/ideal_trace.txt");
     std::vector<Point3D> points;
     if (file.is_open())
     {
          std::string line;
          while (std::getline(file, line))
          {
               std::stringstream ss(line);
               std::string token;
               Point3D point;
               // 使用制表符分隔每行的坐标xyz分量
               std::getline(ss, token, '\t');
               point.x = std::stof(token);
               std::getline(ss, token, '\t');
               point.y = std::stof(token);
               std::getline(ss, token, '\t');
               point.z = std::stof(token);
               points.push_back(point);
          }
          file.close();
     }

     ros::Rate rate(10); // 设置发布频率为30Hz
     geometry_msgs::PointStamped pose_msg;
     pose_msg.header.frame_id = "base_link"; // 设置坐标系为base_link

     for (const auto &point : points)
     {
          pose_msg.header.stamp = ros::Time::now();
          pose_msg.point.x = point.x;
          pose_msg.point.y = point.y;
          pose_msg.point.z = point.z;
          // std::cout << "x: " << pose_msg.point.x << ", y: " << pose_msg.point.y << ", z: " << pose_msg.point.z << std::endl;

          pose_pub.publish(pose_msg);
          ros::spinOnce();
          rate.sleep();
     }

     return 0;
}
