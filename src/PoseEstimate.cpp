//
// Created by flamming on 2018/1/29.
//

#include "PoseEstimate.h"
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>




bool NumericDiffCostFunctor::operator()(const double *vertex, double *residual) const {
    double E = 0;
    residual[0] = 0;
//    for (int i = 0; i < 4; ++i) {
        double x = vertex[0] + X_.x;
        double y = vertex[1] + X_.y;
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



        auto He = (1.0) / ((1) + ceres::exp(-Theta/100));
//
//        E = -ceres::log(He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
//                        + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));

////together with robust function we implement the eassy
        E = -1 + (He * fwd_.at<double>(cv::Point(cvRound(x),cvRound(y)))
                  + (1 - He) * bg_.at<double>(cv::Point(cvRound(x),cvRound(y))));

//        std::cout<<lb<<std::endl;

//        E = He;
        residual[0] += E;

//    }
//    std::cout<<residual[0]<<std::endl;
    return true;
}





