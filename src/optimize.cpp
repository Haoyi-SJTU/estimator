#include <iostream>
#include <gurobi_c++.h>
#include <vector>

using namespace std;

// 视觉惯性里程计的优化问题
class VisualInertialOptimization
{
private:
    GRBEnv env_;              // Gurobi环境
    GRBModel model_;          // Gurobi模型//
    vector<GRBVar> position_; // 位姿和速度
    vector<GRBVar> velocity_;
    vector<GRBVar> quaternion_;          // 四元数
    vector<GRBVar> control_input_;       // 控制输入
    vector<GRBVar> feature_error_;       // 特征点的投影误差
    vector<double> vision_measurements_; // 视觉测量值//
    vector<double> imu_measurements_;    // IMU测量值
    // 每个特征点对应的位姿和速度
    vector<int> feature_pose_indices_;
    vector<int> feature_velocity_indices_;
    vector<int> feature_ids_;
    // 变量的数量
    int num_poses_;//
    int num_controls_;
    int num_features_;//

    double vision_noise_stddev_ = 0.1; // 视觉测量噪声标准差
    double imu_noise_stddev_ = 0.001;  // IMU测量噪声标准差

    double delta_t_ = 0.1; // 模型运动模型参数//

    void AddVariables();   // 添加变量//
    void AddConstraints(); // 添加约束条件//
    void AddObjective();   // 添加目标函数//

public:
    VisualInertialOptimization(int num_poses, int num_controls, int num_features)
        : num_poses_(num_poses), num_controls_(num_controls), num_features_(num_features) {}
    void Solve(); // 解决视觉惯性里程计问题
};

void VisualInertialOptimization::Solve()
{
    AddVariables();
    AddConstraints();
    AddObjective();

    model_.set(GRB_IntParam_Method, GRB_METHOD_BARRIER); // 设置Gurobi参数
    model_.optimize();                                   // 使用gurobi库函数求解

    // 输出结果
    if (model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
    {
        for (int i = 0; i < num_poses_; i++)
        {
            cout << "Pose " << i << ": (";
            cout << position_[3 * i].get(GRB_DoubleAttr_X) << ", ";
            cout << position_[3 * i + 1].get(GRB_DoubleAttr_X) << ", ";
            cout << position_[3 * i + 2].get(GRB_DoubleAttr_X) << ")" << endl;
        }
    }
}

void VisualInertialOptimization::AddVariables()
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
}

void VisualInertialOptimization::AddConstraints()
{
    // 初始位姿和速度
    model_.addConstr(position_[0] == 0.0);
    model_.addConstr(position_[1] == 0.0);
    model_.addConstr(position_[2] == 0.0);
    model_.addConstr(velocity_[0] == 0.0);
    model_.addConstr(velocity_[1] == 0.0);
    model_.addConstr(velocity_[2] == 0.0);

    // 运动模型约束
    for (int i = 0; i < num_poses_ - 1; i++)
    {
        // 计算旋转四元数的下标
        int q_idx = 4 * i;

        // 添加位置运动模型约束
        model_.addConstr(position_[3 * (i + 1)] == position_[3 * i] + delta_t_ * velocity_[3 * i] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i] + quaternion_[q_idx + 2] * control_input_[6 * i + 1] + quaternion_[q_idx + 3] * control_input_[6 * i + 2]));
        model_.addConstr(position_[3 * (i + 1) + 1] == position_[3 * i + 1] + delta_t_ * velocity_[3 * i + 1] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i + 1] - quaternion_[q_idx] * control_input_[6 * i + 2] + quaternion_[q_idx + 3] * control_input_[6 * i + 3]));
        model_.addConstr(position_[3 * (i + 1) + 2] == position_[3 * i + 2] + delta_t_ * velocity_[3 * i + 2] + 0.5 * pow(delta_t_, 2) * (quaternion_[q_idx + 1] * control_input_[6 * i + 2] + quaternion_[q_idx] * control_input_[6 * i + 1] + quaternion_[q_idx + 2] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 4]));
        // 添加速度运动模型约束
        model_.addConstr(velocity_[3 * (i + 1)] == velocity_[3 * i] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 3] + quaternion_[q_idx + 2] * control_input_[6 * i + 4] - quaternion_[q_idx + 3] * control_input_[6 * i + 5]));
        model_.addConstr(velocity_[3 * (i + 1) + 1] == velocity_[3 * i + 1] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 4] - quaternion_[q_idx] * control_input_[6 * i + 5] + quaternion_[q_idx + 2] * control_input_[6 * i + 3]));
        model_.addConstr(velocity_[3 * (i + 1) + 2] == velocity_[3 * i + 2] + delta_t_ * (quaternion_[q_idx + 1] * control_input_[6 * i + 5] + quaternion_[q_idx] * control_input_[6 * i + 4] - quaternion_[q_idx + 2] * control_input_[6 * i + 3]));
        // 添加四元数运动模型约束
        model_.addConstr(quaternion_[q_idx] * quaternion_[q_idx] + quaternion_[q_idx + 1] * quaternion_[q_idx + 1] + quaternion_[q_idx + 2] * quaternion_[q_idx + 2] + quaternion_[q_idx + 3] * quaternion_[q_idx + 3] == 1.0);
        model_.addConstr(quaternion_[q_idx] == quaternion_[q_idx + 3] * control_input_[6 * i] - quaternion_[q_idx + 2] * control_input_[6 * i + 1] + quaternion_[q_idx + 1] * control_input_[6 * i + 2]);
        model_.addConstr(quaternion_[q_idx + 1] == quaternion_[q_idx + 2] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 1] - quaternion_[q_idx] * control_input_[6 * i + 2]);
        model_.addConstr(quaternion_[q_idx + 2] == -quaternion_[q_idx + 1] * control_input_[6 * i] + quaternion_[q_idx + 3] * control_input_[6 * i + 2] + quaternion_[q_idx] * control_input_[6 * i + 1]);
        model_.addConstr(quaternion_[q_idx + 3] == -quaternion_[q_idx] * control_input_[6 * i] - quaternion_[q_idx + 1] * control_input_[6 * i + 1] - quaternion_[q_idx + 2] * control_input_[6 * i + 2]);
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
            model_.addConstr(feature_pose_indices_[i] >= j - (num_poses_ - 1) * (1 - binary_pose_vars_[j][i]));
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

    // 目标函数: 最小化投影误差和控制力量的平方和
    GRBQuadExpr obj = 0.0;
    for (int i = 0; i < num_features_; i++)
    {
        obj += feature_error_[2 * i] + feature_error_[2 * i + 1];
    }

    for (int i = 0; i < num_poses_ - 1; i++)
    {
        obj += control_input_[6 * i] * control_input_[6 * i] + control_input_[6 * i + 1] * control_input_[6 * i + 1] + control_input_[6 * i + 2] * control_input_[6 * i + 2] + control_input_[6 * i + 3] * control_input_[6 * i + 3] + control_input_[6 * i + 4] * control_input_[6 * i + 4] + control_input_[6 * i + 5] * control_input_[6 * i + 5];
    }

    model_.setObjective(obj);
}

