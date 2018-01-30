//
// Created by flamming on 2018/1/30.
//
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#ifndef HISTOGRAMTRACKER_MODEL_H
#define HISTOGRAMTRACKER_MODEL_H

class Model
{
public:
    Model();
    static Model* GetInstance(){
        static Model* m = new Model;
        m->SetIntrinsic();
        return m;
    }
    void SetIntrinsic();
    void Display(cv::Mat& canvas,Sophus::SE3d pose);
    cv::Mat DrawMask(cv::Mat& mask,Sophus::SE3d pose);
    cv::Point2d Project2d(cv::Point3d vertex,Sophus::SE3d pose);

    cv::Matx33d intrinsic_;
    std::vector<cv::Point3d> vertex_;
};

#endif //HISTOGRAMTRACKER_MODEL_H
