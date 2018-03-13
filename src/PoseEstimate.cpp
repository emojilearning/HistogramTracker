//
// Created by flamming on 2018/1/29.
//

#include "PoseEstimate.h"
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include "Model.h"
#include "GlobalConfig.h"

using namespace cv;
using namespace Sophus;
using namespace std;


bool NumericDiffCostFunctor::operator()(const double *pose, double *residual) const {
    residual[0] = 0;
    Eigen::Vector3d tandr(pose[3],pose[4],pose[5]);
    Eigen::Vector3d rot(0,0,0);//pose[0],pose[1],pose[2]);
    SE3d pose_t(SO3d::exp(rot),tandr);

    Point2d p = Model::GetInstance()->Project2d(X_,pose_t);

    double x = p.x+offset_.x;
    double y = p.y+offset_.y;

    //bilinear interpolation
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

/// ///together with robust function we implement the eassy.considering that ceres-solver using min-square the "-" is dropped

    E = -1 +
            (He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
              + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));



//        E = Theta;
    residual[0] = E;

//    }
//    std::cout<<residual[0]<<std::endl;
    return true;
}

//
//bool CostFunctionByJac::Evaluate(double const * const *parameters,
//                      double *residuals,
//                      double **jacobians) const {
//    static int cnt = 0;
//    if (!jacobians)
//    {
//        cout<<cnt++<<endl;
//        return true;
//    }
//    double *jacobian = jacobians[0];
//    if (!jacobian) return true;
//    const double *pose = parameters[0];
//    for (int i = 0; i < 6; i++) {
//        jacobians[0][i] = 1;
//    }
//    residuals[0] = 1;
//    return true;
//
//    Eigen::Vector3d tandr(pose[3],pose[4],pose[5]);
//    Eigen::Vector3d rot(pose[0],pose[1],pose[2]);
//    SE3d pose_t(SO3d::exp(rot),tandr);
//
//
//    Point2d p = Model::GetInstance()->Project2d(X_,pose_t);
//
//    double x = p.x + offset_.x;
//    double y = p.y + offset_.y;
//
//    //bilinear interpolation
//    int xf = (int)floor(x);
//    int yf = (int)floor(y);
//    int xc = (int)ceil(x);
//    int yc = (int)ceil(y);
//
//    auto lb = dt_map_.at<float>(cv::Point(xf,yf));
//    auto rb = dt_map_.at<float>(cv::Point(xc,yf));
//    auto lt = dt_map_.at<float>(cv::Point(xf,yc));
//    auto rt = dt_map_.at<float>(cv::Point(xc,yc));
//
//    auto mb = (lb + (x - xf)*(rb - lb));
//    auto mt = (lt + (x - xf)*(rt - lt));
//
//    auto dt = mb + (mt - mb)*(y - yf);
//    double Theta = (dt);
//
//    //theta here is wired...
//    auto He = (1.0) / ((1) + ceres::exp(-Theta));
//
//    double E;
////
////        E = -ceres::log(He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
////                        + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));
//
////////together with robust function we implement the eassy.considering that ceres-solver using min-square the "-" is dropped
//
//    E = -1 +
//        (He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
//         + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));
//
//
//
////        E = Theta;
//    residuals[0] = E;
//
//
//    double left = (1 - 1/(1+Theta)) * (fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
//                                       - bg_.at<double>(cv::Point(cvRound(x),cvRound(y)))) /
//                  (He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
//                   + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));
//
//
//    Eigen::MatrixXd j_X_Lie(2, 6);
//    Eigen::MatrixXd j_Phi_x(1, 2);
//
//    auto model_instance = Model::GetInstance();
//    auto x_plane = p;
//
//    auto X_Camera_coord = pose_t * Vector3d(X_.x,X_.y,X_.z);
//    double _x_in_Camera = X_Camera_coord[0];
//    double _y_in_Camera = X_Camera_coord[1];
//    double _z_in_Camera = X_Camera_coord[2];
//
//
////    if(!fabs((He * fwd_.at<double>(x_plane) + (1 - He) * bg_.at<double>(x_plane))>0.1)) {
////        _x_in_Camera = 1;
////        std::cout<<x_plane<<std::endl;
////    }
////        assert(fabs((He * fwd_.at<double>(x_plane) + (1 - He) * bg_.at<double>(x_plane)))>0.001);
//
//
//    auto gConfig = Config::configInstance();
//    j_X_Lie(0, 0) = -gConfig.FX / _z_in_Camera;
//    j_X_Lie(0, 1) = 0;
//    j_X_Lie(0, 2) = gConfig.FX * _x_in_Camera / (_z_in_Camera * _z_in_Camera);
//    j_X_Lie(0, 3) = gConfig.FX * _x_in_Camera * _y_in_Camera / (_z_in_Camera * _z_in_Camera);
//    j_X_Lie(0, 4) = -gConfig.FX * (1 + _x_in_Camera * _x_in_Camera / (_z_in_Camera * _z_in_Camera));
//    j_X_Lie(0, 5) = gConfig.FX * _y_in_Camera / _z_in_Camera;
//
//    j_X_Lie(1, 0) = 0;
//    j_X_Lie(1, 1) = -gConfig.FY / _z_in_Camera;
//    j_X_Lie(1, 2) = gConfig.FY * _y_in_Camera / (_z_in_Camera * _z_in_Camera);
//    j_X_Lie(1, 3) = gConfig.FY * (1 + _y_in_Camera * _y_in_Camera * (1 / (_z_in_Camera * _z_in_Camera)));
//    j_X_Lie(1, 4) = -gConfig.FY * _x_in_Camera * _y_in_Camera / (_z_in_Camera * _z_in_Camera);
//    j_X_Lie(1, 5) = -gConfig.FY * _x_in_Camera / _z_in_Camera;
//
//    j_Phi_x(0, 0) = 0.5f * (dt_map_.at<float>(cv::Point(x_plane.x + 1, x_plane.y)) -
//                            dt_map_.at<float>(cv::Point(x_plane.x - 1, x_plane.y)));
//    j_Phi_x(0, 1) = 0.5f * (dt_map_.at<float>(cv::Point(x_plane.x, x_plane.y + 1)) -
//                            dt_map_.at<float>(cv::Point(x_plane.x, x_plane.y - 1)));
//    Eigen::MatrixXd jac = j_Phi_x * j_X_Lie;
//
//    for (int i = 0; i < 6; i++) {
//        jacobian[i] = jac(0, i);
//    }
//
//    return true;
//}
//
//
//
//
