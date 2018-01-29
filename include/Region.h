//
// Created by flamming on 2018/1/14.
//

#ifndef ROBOT_REGION_H
#define ROBOT_REGION_H

#include <opencv2/opencv.hpp>


void BresenhamCircle(cv::Point2i veterx,int radius,std::vector<cv::Point2i>& sampleset);
void BresenhamCircle(int radius,std::vector<cv::Point2i>& sampleset);

//class Region {
//public:
//    Region(){};
//    Region(cv::Point center,int r);
//    cv::Point center_;
//    int radius_;
//    double n_fw{};
//    double n_bg{};
//    double aera;
//
//
//    void UpdateHistorgram(FramePtr curFrame);
//    void UpdateHistorgram(Frame* curFrame);
//    void VizHistImg(const Histogram& img);
//    std::vector<cv::Point> circle_bound_;
//};


#endif //ROBOT_REGION_H
