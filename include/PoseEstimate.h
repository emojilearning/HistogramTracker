//
// Created by flamming on 2018/1/29.
//

#ifndef HISTOGRAMTRACKER_SOLVER_H
#define HISTOGRAMTRACKER_SOLVER_H

#include <opencv2/opencv.hpp>

struct NumericDiffCostFunctor
{
    NumericDiffCostFunctor(cv::Point3d &X,cv::Point offset,cv::Mat &fwd, cv::Mat &bg, cv::Mat &dt_map) :
            X_(X), offset_(offset), fwd_(fwd), bg_(bg), dt_map_(dt_map) {};
    bool operator()(const double *pose, double *residual) const;

    const cv::Point3d X_;
    const cv::Mat dt_map_;
    const cv::Mat fwd_;
    const cv::Mat bg_;
    const cv::Point offset_;

};

#endif //HISTOGRAMTRACKER_SOLVER_H
