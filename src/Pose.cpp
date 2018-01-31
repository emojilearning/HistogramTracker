#include "Pose.h"

using namespace std;

Sophus::SE3d Data2Pose(double * data)
{
    Sophus::Vector6d se3_pose;
    se3_pose<<(double)data[3],(double)data[4],(double)data[5],
            (double)data[0],(double)data[1],(double)data[2];

    auto m_pose = Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d((double) data[0], (double) data[1], (double) data[2])),
                               Eigen::Vector3d((double) data[3], (double) data[4], (double) data[5]));

    return m_pose;
}

Sophus::SE3d Data2Pose(float * data)
{
    Sophus::Vector6d se3_pose;
    se3_pose<<(double)data[3],(double)data[4],(double)data[5],
            (double)data[0],(double)data[1],(double)data[2];

    auto m_pose = Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d((double) data[0], (double) data[1], (double) data[2])),
                          Eigen::Vector3d((double) data[3], (double) data[4], (double) data[5]));

    return m_pose;
}

cv::Mat Se2cvf(Sophus::SE3d pose)
{
    cv::Mat temp;
    cv::eigen2cv(pose.matrix(),temp);
    return temp;
}
