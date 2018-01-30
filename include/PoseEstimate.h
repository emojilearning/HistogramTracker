//
// Created by flamming on 2018/1/29.
//

#ifndef HISTOGRAMTRACKER_SOLVER_H
#define HISTOGRAMTRACKER_SOLVER_H

#include <opencv2/opencv.hpp>

struct NumericDiffCostFunctor
{
    NumericDiffCostFunctor(cv::Point &X, cv::Mat &fwd, cv::Mat &bg, cv::Mat &dt_map) :
            X_(X), fwd_(fwd), bg_(bg), dt_map_(dt_map) {};
    bool operator()(const double *vertex, double *residual) const;

    const cv::Point X_;
    const cv::Mat dt_map_;
    const cv::Mat fwd_;
    const cv::Mat bg_;

};

#endif //HISTOGRAMTRACKER_SOLVER_H
