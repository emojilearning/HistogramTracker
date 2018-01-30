//
// Created by flamming on 2018/1/29.
//

#include "PoseEstimate.h"
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include "Model.h"

using namespace cv;
using namespace Sophus;


bool NumericDiffCostFunctor::operator()(const double *pose, double *residual) const {
    residual[0] = 0;
    Eigen::Vector3d tandr(pose[3],pose[4],pose[5]);
    Eigen::Vector3d rot(pose[0],pose[1],pose[2]);
    SE3d pose_t(SO3d::exp(rot),tandr);

    Point2d p = Model::GetInstance()->Project2d(X_,pose_t);

    double x = p.x;
    double y = p.y;
    int xf = (int)floor(x);
    int yf = (int)floor(y);
    int xc = (int)ceil(x);
    int yc = (int)ceil(y);

    auto lb = dt_map_.at<float>(cv::Point(xf,yf));
    auto rb = dt_map_.at<float>(cv::Point(xc,yf));
    auto lt = dt_map_.at<float>(cv::Point(xf,yc));
    auto rt = dt_map_.at<float>(cv::Point(xc,yc));

    auto mb = (lb + (x - xf)*(rb - lb));
    auto mt = (lt + (x - xf)*(rt - lt));

    auto dt = mb + (mt - mb)*(y - yf);
    double Theta = (dt);

    //theta here is wired...
    auto He = (1.0) / ((1) + ceres::exp(-Theta));

    double E;
//
//        E = -ceres::log(He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
//                        + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));

////together with robust function we implement the eassy
    E = -1 + (He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
              + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));

//        E = He;
    residual[0] = Theta;

//    }
//    std::cout<<residual[0]<<std::endl;
    return true;
}





