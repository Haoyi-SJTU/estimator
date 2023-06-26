#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <vector>

using namespace std;
// 定义优化目标函数
void vio_optimization(const std::vector<float> &imu_acc, const std::vector<float> &s, GRBModel &model, float *position, float *velocity)
{
    int n = imu_acc.size(); // IMU数据点数
    float dt = 1;           // 时间步长
    GRBVar x[n];            // 机器人位姿
    GRBVar v[n];            // 机器人位姿

    // 定义优化变量属性：第三个参数表示该变量的系数（0表示不参与目标函数的计算），GRB_CONTINUOUS表示该变量为连续型变量
    for (int i = 0; i < n; i++)
    {
        x[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
        v[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
    }
    // 添加约束条件
    for (int i = 0; i < n - 1; i++)
    {
        // 约束机器人位姿的变化模型
        model.addConstr(x[i + 1] == x[i] + v[i] * dt + 0.5 * dt * dt * imu_acc[i]);
        model.addConstr(v[i + 1] == v[i] + dt * imu_acc[i]);
    }

    GRBQuadExpr obj; // 定义目标函数（最小化机器人位姿和特征点位置的位置估计误差）

    for (int i = 0; i < n - 1; i++)
        obj += (x[i] - s[i]) * (x[i] - s[i]);
    try
    {
        model.setObjective(obj);
        model.optimize();
        for (int i = 0; i < n; i++)
            position[i] = x[i].get(GRB_DoubleAttr_X);
        for (int j = 0; j < n; j++)
            velocity[j] = v[j].get(GRB_DoubleAttr_X);
    }
    catch (GRBException &ex)
    {
        cout << "Error code =" << ex.getErrorCode() << endl;
        cout << ex.getMessage() << endl;
    }
}

int main()
{
    srand(time(NULL));
    int n = 20; // IMU数据点数
    float dt = 1;
    float acc = 1;
    // 生成IMU加速度测量值
    std::vector<float> acc_test(n);
    std::vector<float> s(n);
    std::vector<float> destina_s(s);
    std::vector<float> v(n);
    for (int i = 0; i < n - 1; i++)
    {
        float random = (rand() % 1000);
        acc_test[i] = acc + random / 5000;
        float random2 = (rand() % 1000);
        v[i + 1] = v[i] + dt * acc;
        destina_s[i + 1] = destina_s[i] + v[i] * dt + 0.5 * dt * dt * acc;
        s[i + 1] = abs(s[i] + v[i] * dt + 0.5 * dt * dt * acc - random2 / 5000);
    }

    for (int i = 0; i < n; i++) // 打印结果
        std::cout << "s[" << i << "] = " << s[i] << std::endl;

    // 创建优化模型
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    float position[n];
    float velocity[n];
    vio_optimization(acc_test, s, model, position, velocity); // 进行优化求解

    for (int i = 0; i < n; i++) // 打印结果
    {
        std::cout << "destina pos[" << i << "] = " << destina_s[i] << "\t \t";
        std::cout << "optimize pos = " << position[i] << std::endl;
    }
    for (int i = 0; i < n; i++) // 打印结果
    {
        std::cout << "velocity[" << i << "] after optimize = " << velocity[i] << std::endl;
    }

    return 0;
}