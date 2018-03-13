//
// Created by flamming on 2018/1/29.
//

#ifndef HISTOGRAMTRACKER_SOLVER_H
#define HISTOGRAMTRACKER_SOLVER_H

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

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


class CostFunctionByJac : public ceres::SizedCostFunction<1, 6> {

public:
    CostFunctionByJac(const cv::Point3d& X, const cv::Mat &fwd, const cv::Mat &bg, const cv::Mat &dt_map, const cv::Point& offset) :
            X_(X), fwd_(fwd),offset_(offset), bg_(bg), dt_map_(dt_map) {}

    virtual ~CostFunctionByJac(){}

    virtual bool Evaluate(double const * const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        static int cnt = 0;
        if (!jacobians)
        {
            std::cout<<cnt++<<std::endl;
            return true;
        }
        double *jacobian = jacobians[0];
        if (!jacobian) return true;
        const double *pose = parameters[0];
        for (int i = 0; i < 6; i++) {
            jacobians[0][i] = 1;
        }
        residuals[0] = 1;
        return true;
    }

    const cv::Point3d X_;
    const cv::Mat dt_map_;
    const cv::Mat fwd_;
    const cv::Mat bg_;
    const cv::Point offset_;
};


#endif //HISTOGRAMTRACKER_SOLVER_H
