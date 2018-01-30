//
// Created by flamming on 2018/1/29.
//
#ifndef HISTOGRAMTRACKER_VIZ_H
#define HISTOGRAMTRACKER_VIZ_H

#include <opencv2/opencv.hpp>

void draw(cv::Mat& out1, int x_before,int y_before,int l_x_b,int l_y_b);
void draw(cv::Mat& out1, std::vector<cv::Point2d> p,cv::Scalar color={0,255,0});

#endif //HISTOGRAMTRACKER_VIZ_H
