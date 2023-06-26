#include <iostream>
#include <gurobi_c++.h>
#include <vector>

using namespace std;

class SlamOptimizer
{
private:
    GRBModel model_; // Gurobi模型

    // 变量的数量
    int num_poses_;
    int num_features_;

    vector<double> vision_measurements_;   // 视觉测量值
    vector<double> control_input_;         // 控制输入
    vector<GRBVar> position_;              // 位姿
    vector<GRBVar> velocity_;              // 速度
    vector<GRBVar> quaternion_;            // 四元数
    vector<GRBVar> control_input_;         // 控制输入
    vector<GRBVar> feature_error_;         // 特征点的投影误差
    vector<int> feature_pose_indices_;     // 每个特征点对应的位姿
    vector<int> feature_velocity_indices_; // 每个特征点对应的速度
    double delta_t_;

    void addVariables();   // 添加变量
    void addConstraints(); // 添加约束条件
    void addObjective();   // 添加目标函数
    void writeOptimizedTrajectoryToFile(const std::string &filename) const;
    double computeProjectionError() const;
    double computeControlEnergy() const;

public:
    struct OptimizationResult
    {
        Trajectory optimized_trajectory;
        Trajectory optimized_trajectory_velocities;
        double projection_error;
        double control_energy;
    };

    SlamOptimizer();
    OptimizationResult optimize();
};

SlamOptimizer::SlamOptimizer()
{
    // num_poses
    // num_features
    // vision_measurements
    // control_input
    // feature_pose_indices
    // feature_velocity_indices
    delta_t = 0.01;

    addVariables();
    addConstraints();
    addObjective();
}

// 调用 Gurobi 求解器，并返回优化后的状态和错误指标
OptimizationResult SlamOptimizer::optimize()
{
    OptimizationResult result;
    OptimizationResult result = optimizer.optimize();

    std::cout << "Optimization complete. \n " << std::endl;
    std::cout << "Projection error: " << result.projection_error << std::endl;
    std::cout << "Control energy: " << result.control_energy << std::endl;
    try
    {
        model_.optimize();

        if (model_.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
        {
            throw std::runtime_error("ERROR: No an optimal solution");
        }

        result.optimized_trajectory = Trajectory(num_poses_);
        result.optimized_trajectory_velocities = Trajectory(num_poses_);

        for (int i = 0; i < num_poses_; i++)
        {
            result.optimized_trajectory(i, 0) = model_.getVarByName("position_x_" + std::to_string(i)).get(GRB_DoubleAttr_X);
            result.optimized_trajectory(i, 1) = model_.getVarByName("position_y_" + std::to_string(i)).get(GRB_DoubleAttr_X);
            result.optimized_trajectory(i, 2) = model_.getVarByName("position_z_" + std::to_string(i)).get(GRB_DoubleAttr_X);
            result.optimized_trajectory_velocities(i, 0) = model_.getVarByName("velocity_x_" + std::to_string(i)).get(GRB_DoubleAttr_X);
            result.optimized_trajectory_velocities(i, 1) = model_.getVarByName("velocity_y_" + std::to_string(i)).get(GRB_DoubleAttr_X);
            result.optimized_trajectory_velocities(i, 2) = model_.getVarByName("velocity_z_" + std::to_string(i)).get(GRB_DoubleAttr_X);
        }

        result.projection_error = computeProjectionError();
        result.control_energy = computeControlEnergy();
    }
    catch (GRBException &e)
    {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        throw std::runtime_error("Gurobi optimizer error occurred");
    }

    return result;
}

// 添加优化变量
void SlamOptimizer::addVariables()
{
    // 位姿、速度和四元数
    position_.resize(3 * num_poses_);
    velocity_.resize(3 * num_poses_);
    quaternion_.resize(4 * num_poses_);
    for (int i = 0; i < num_poses_; i++)
    {
        // 添加位姿变量
        position_[3 * i] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        position_[3 * i + 1] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        position_[3 * i + 2] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        // 添加速度变量
        velocity_[3 * i] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        velocity_[3 * i + 1] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        velocity_[3 * i + 2] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        // 添加四元数变量
        quaternion_[4 * i] = model_.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
        quaternion_[4 * i + 1] = model_.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
        quaternion_[4 * i + 2] = model_.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
        quaternion_[4 * i + 3] = model_.addVar(-1.0, 1.0, 0.0, GRB_CONTINUOUS);
    }

    // 控制输入
    control_input_.resize(6 * num_controls_);
    for (int i = 0; i < num_controls_; i++)
    {
        control_input_[6 * i] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        control_input_[6 * i + 1] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        control_input_[6 * i + 2] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        control_input_[6 * i + 3] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        control_input_[6 * i + 4] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        control_input_[6 * i + 5] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }

    // 特征点的投影误差
    feature_error_.resize(2 * num_features_);
    for (int i = 0; i < num_features_; i++)
    {
        feature_error_[2 * i] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        feature_error_[2 * i + 1] = model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }

    for (int i = 0; i < num_features_; i++)
    {
        model_.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "inv_depth_" + std::to_string(i));
    }
}

// 添加目标函数
void SlamOptimizer::addObjective()
{
    GRBQuadExpr objective;

    // 目标函数: 最小化投影误差和控制力量的平方和
    GRBQuadExpr objective = 0;
    for (int i = 0; i < num_features_; i++)
    {
        objective += feature_error_[2 * i] + feature_error_[2 * i + 1];
    }

    // 位姿控制成本
    for (int i = 0; i < num_poses_ - 1; i++)
    {
        objective += pow(model_.getVarByName("velocity_x_" + std::to_string(i)), 2) +
                     pow(model_.getVarByName("velocity_y_" + std::to_string(i)), 2) +
                     pow(model_.getVarByName("velocity_z_" + std::to_string(i)), 2);
    }

    // 特征点重投影误差成本
    for (int i = 0; i < num_features_; i++)
    {
        int pose_index = feature_pose_indices_[i];
        int velocity_index = feature_velocity_indices_[i];

        double depth_guess = 1.0;
        if (pose_index > 0)
        {
            depth_guess = 1.0 / (model_.getVarByName("inv_depth_" + std::to_string(i)).get(GRB_DoubleAttr_Start) +
                                 model_.getVarByName("velocity_z_" + std::to_string(velocity_index)).get(GRB_DoubleAttr_Start) * delta_t_);
        }

        double measured = vision_measurements_[i];

        objective += pow(measured - depth_guess * ((model_.getVarByName("position_x_" + std::to_string(pose_index)) -
                                                    model_.getVarByName("position_x_" + std::to_string(num_poses_ - 1))) *
                                                       cos(model_.getVarByName("position_z_" + std::to_string(pose_index))) +
                                                   (model_.getVarByName("position_y_" + std::to_string(pose_index)) -
                                                    model_.getVarByName("position_y_" + std::to_string(num_poses_ - 1))) *
                                                       sin(model_.getVarByName("position_z_" + std::to_string(pose_index)))),
                         2);
    }
    model_.setObjective(objective, GRB_MINIMIZE);
}

// 添加约束条件
void SlamOptimizer::addConstraints()
{
    // 初始位姿和速度
    model_.addConstr(position_[0] == 0.0);
    model_.addConstr(position_[1] == 0.0);
    model_.addConstr(position_[2] == 0.0);
    model_.addConstr(velocity_[0] == 0.0);
    model_.addConstr(velocity_[1] == 0.0);
    model_.addConstr(velocity_[2] == 0.0);

    // 模型约束
    for (int i = 1; i < num_poses_; i++)
    {
        int q_idx = 4 * i; // 计算旋转四元数的下标
        // 位置约束
        model_.addConstr(position_[3 * (i + 1)] == position_[3 * i] + delta_t_ * velocity_[3 * i] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i] + quaternion_[q_idx + 2] * control_input_[6 * i + 1] + quaternion_[q_idx + 3] * control_input_[6 * i + 2]));
        model_.addConstr(position_[3 * (i + 1) + 1] == position_[3 * i + 1] + delta_t_ * velocity_[3 * i + 1] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i + 1] - quaternion_[q_idx] * control_input_[6 * i + 2] + quaternion_[q_idx + 3] * control_input_[6 * i + 3]));
        model_.addConstr(position_[3 * (i + 1) + 2] == position_[3 * i + 2] + delta_t_ * velocity_[3 * i + 2] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i + 2] + quaternion_[q_idx] * control_input_[6 * i + 1] + quaternion_[q_idx + 2] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 4]));
        // 速度约束
        model_.addConstr(velocity_[3 * (i + 1)] == velocity_[3 * i] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 3] + quaternion_[q_idx + 2] * control_input_[6 * i + 4] - quaternion_[q_idx + 3] * control_input_[6 * i + 5]));
        model_.addConstr(velocity_[3 * (i + 1) + 1] == velocity_[3 * i + 1] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 4] - quaternion_[q_idx] * control_input_[6 * i + 5] + quaternion_[q_idx + 2] * control_input_[6 * i + 3]));
        model_.addConstr(velocity_[3 * (i + 1) + 2] == velocity_[3 * i + 2] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 5] + quaternion_[q_idx] * control_input_[6 * i + 4] - quaternion_[q_idx + 2] * control_input_[6 * i + 3]));
        // 转角四元数约束
        model_.addConstr(quaternion_[q_idx] * quaternion_[q_idx] + quaternion_[q_idx + 1] * quaternion_[q_idx + 1] + quaternion_[q_idx + 2] * quaternion_[q_idx + 2] + quaternion_[q_idx + 3] * quaternion_[q_idx + 3] == 1.0);
        model_.addConstr(quaternion_[q_idx] == quaternion_[q_idx + 3] * control_input_[6 * i] - quaternion_[q_idx + 2] * control_input_[6 * i + 1] + quaternion_[q_idx + 1] * control_input_[6 * i + 2]);
        model_.addConstr(quaternion_[q_idx + 1] == quaternion_[q_idx + 2] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 1] - quaternion_[q_idx] * control_input_[6 * i + 2]);
        model_.addConstr(quaternion_[q_idx + 2] == -quaternion_[q_idx + 1] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 2] + quaternion_[q_idx] * control_input_[6 * i + 1]);
        model_.addConstr(quaternion_[q_idx + 3] == -quaternion_[q_idx] * control_input_[6 * i] - quaternion_[q_idx + 1] * control_input_[6 * i + 1] - quaternion_[q_idx + 2] * control_input_[6 * i + 2]);                                                                           control_input_[6 * (i - 1) + 5]);
    }

    // 特征点对应的位姿和速度feature_pose_indices_.resize(num_features_);feature_velocity_indices_.resize(num_features_);feature_ids_.resize(num_features_);
    for (int i = 0; i < num_features_; i++)
    { // 添加特征点对应的位姿和速度变量feature_pose_indices_[i] = model_.addVar(0.0, num_poses_ - 1, 0.0, GRB_INTEGER);feature_velocity_indices_[i] = model_.addVar(0.0, num_poses_ - 1, 0.0, GRB_INTEGER);
        // 设置特征点ID
        feature_ids_[i] = i;

        // 添加特征点投影误差约束
        model_.addConstr(feature_error_[2 * i] == 0.0);
        model_.addConstr(feature_error_[2 * i + 1] == 0.0);

        for (int j = 0; j < num_poses_; j++)
        {
            // 添加特征点对应的位姿和速度的约束
            model_.addConstr(feature_pose_indices_[i] >= j - (num_poses-1) * (1-binary_pose_vars_[j][i]));
            model_.addConstr(feature_pose_indices_[i] <= j + (num_poses_ - 1) * (1 - binary_pose_vars_[j][i]));
            model_.addConstr(feature_velocity_indices_[i] >= j - (num_poses_ - 1) * (1 - binary_velocity_vars_[j][i]));
            model_.addConstr(feature_velocity_indices_[i] <= j + (num_poses_ - 1) * (1 - binary_velocity_vars_[j][i]));
            // 添加特征点投影误差
            model_.addConstr(feature_error_[2 * i] += (position_[3 * j] - vision_measurements_[6 * i] * position_[3 * feature_pose_indices_[i]] - vision_measurements_[6 * i + 1]) / vision_measurements_[6 * i + 5] * pixel_noise_stddev_ * pixel_noise_stddev_);
            model_.addConstr(feature_error_[2 * i + 1] += (position_[3 * j + 1] - vision_measurements_[6 * i + 2] * position_[3 * feature_pose_indices_[i] + 1] - vision_measurements_[6 * i + 3]) / vision_measurements_[6 * i + 6] * pixel_noise_stddev_ * pixel_noise_stddev_);
            // 添加特征点对应的位姿和速度的选择约束
            model_.addConstr(binary_pose_vars_[j][i] + binary_velocity_vars_[j][i] <= 1);
        }
    }

    // // 添加特征点测量约束
    // for (int i = 0; i < num_features_; i++)
    // {
    //     int pose_index = feature_pose_indices_[i];
    //     int velocity_index = feature_velocity_indices_[i];
    //     double depth_guess = 1.0;
    //     if (pose_index > 0)
    //     {
    //         depth_guess = 1.0 / (model_.getVarByName("inv_depth_" + std::to_string(i)).get(GRB_DoubleAttr_Start) +
    //                              model_.getVarByName("velocity_z_" + std::to_string(velocity_index)).get(GRB_DoubleAttr_Start) * delta_t_);
    //     }
    //     double measured = vision_measurements_[i];
    //     model_.addConstr(measured == depth_guess * ((model_.getVarByName("position_x_" + std::to_string(pose_index)) -
    //                                                  model_.getVarByName("position_x_" + std::to_string(num_poses_ - 1))) *
    //                                                     cos(model_.getVarByName("position_z_" + std::to_string(pose_index))) +
    //                                                 (model_.getVarByName("position_y_" + std::to_string(pose_index)) -
    //                                                  model_.getVarByName("position_y_" + std::to_string(num_poses_ - 1))) *
    //                                                     sin(model_.getVarByName("position_z_" + std::to_string(pose_index)))));
    // }
}

// 解析求解结果，提取优化后的特征点信息
std::vectorEigen::VectorXd SlamOptimizer::getOptimizedFeatures() const
{
    std::vector<Eigen::VectorXd> features(num_features_);

    for (int i = 0; i < num_features_; i++)
    {
        features[i].resize(3);

        int pose_index = static_cast<int>(model_.getVarByName("feature_pose_" + std::to_string(i)).get(GRB_DoubleAttr_X));
        int velocity_index = static_cast<int>(model_.getVarByName("feature_velocity_" + std::to_string(i)).get(GRB_DoubleAttr_X));

        features[i][0] = model_.getVarByName("position_x_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X);
        features[i][1] = model_.getVarByName("position_y_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X);
        features[i][2] = model_.getVarByName("position_z_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X) + model_.getVarByName("velocity_z_" + std::to_string(velocity_index)).get(GRB_DoubleAttr_X) * delta_t_;
    }

    return features;
}

// 下面是一些辅助函数的实现，这些函数用于将优化后的位姿和特征点信息保存到文件中，以及计算投影误差等相关指标：
void SlamOptimizer::writeOptimizedTrajectoryToFile(const std::string &filename) const
{
    std::ofstream file(filename); // 将优化后的位姿和速度信息保存到文件中
    for (int i = 0; i < num_poses_; i++)
    {
        file << model_.getVarByName("position_x_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << model_.getVarByName("position_y_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << model_.getVarByName("position_z_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << model_.getVarByName("velocity_x_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << model_.getVarByName("velocity_y_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << model_.getVarByName("velocity_z_" + std::to_string(i)).get(GRB_DoubleAttr_X) << " "
             << std::endl;
    }
    file.close();
}

// 计算投影误差
double SlamOptimizer::computeProjectionError() const
{
    double projection_error = 0.0;

    for (int i = 0; i < num_features_; i++)
    {
        int pose_index = feature_pose_indices_[i];
        int velocity_index = feature_velocity_indices_[i];

        double depth_guess = 1.0;
        if (pose_index > 0)
        {
            depth_guess = 1.0 / (model_.getVarByName("inv_depth_" + std::to_string(i)).get(GRB_DoubleAttr_X) +
                                 model_.getVarByName("velocity_z_" + std::to_string(velocity_index)).get(GRB_DoubleAttr_X) * delta_t_);
        }

        double measured = vision_measurements_[i];
        double predicted = depth_guess * ((model_.getVarByName("position_x_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X) -
                                           model_.getVarByName("position_x_" + std::to_string(num_poses_ - 1)).get(GRB_DoubleAttr_X)) *
                                              cos(model_.getVarByName("position_z_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X)) +
                                          (model_.getVarByName("position_y_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X) -
                                           model_.getVarByName("position_y_" + std::to_string(num_poses_ - 1)).get(GRB_DoubleAttr_X)) *
                                              sin(model_.getVarByName("position_z_" + std::to_string(pose_index)).get(GRB_DoubleAttr_X)));

        projection_error += pow(measured - predicted, 2);
    }

    return projection_error;
}

// 计算重投影误差的过程如下：
//     定义相机内参矩阵和畸变系数。相机内参矩阵包括焦距、主点坐标和像素尺寸等信息，畸变系数用于校正镜头畸变。
//     对于每个空间特征点，将其从3D空间坐标系转换为相机坐标系。
//     使用相机内参矩阵和畸变系数将相机坐标系中的点映射到图像平面上得到2D特征点的预测位置。
//     计算预测的2D特征点位置与实际2D特征点位置之间的欧氏距离，即为重投影误差。
// 以下是一个示例代码，假设已知空间特征点的三维坐标point3D和对应的2D特征点像素坐标point2D：
cv::Mat cameraMatrix; // 相机内参矩阵
cv::Mat distortionCoeffs; // 畸变系数
// 计算重投影误差
double calculateReprojectionError(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Point2f>& points2D)
{
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(points3D, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distortionCoeffs, projectedPoints);

    double totalError = 0.0;
    for (size_t i = 0; i < points2D.size(); ++i)
    {
        double error = cv::norm(projectedPoints[i] - points2D[i]);
        totalError += error;
    }
    return totalError / points2D.size();
}

int main()
{
    // 假设已知空间特征点的三维坐标和对应的2D特征点像素坐标
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;

    // 进行相机内参矩阵和畸变系数的初始化

    double reprojectionError = calculateReprojectionError(points3D, points2D);
    std::cout << "Reprojection Error: " << reprojectionError << std::endl;

    return 0;
}


// 计算控制能量
double SlamOptimizer::computeControlEnergy() const
{
    double control_energy = 0.0;

    for (int i = 0; i < num_poses_ - 1; i++)
    {
        control_energy += pow(model_.getVarByName("velocity_x_" + std::to_string(i)).get(GRB_DoubleAttr_X), 2) +
                          pow(model_.getVarByName("velocity_y_" + std::to_string(i)).get(GRB_DoubleAttr_X), 2) +
                          pow(model_.getVarByName("velocity_z_" + std::to_string(i)).get(GRB_DoubleAttr_X), 2);
    }

    return control_energy;
}

