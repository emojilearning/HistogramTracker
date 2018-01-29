//
// Created by flamming on 2018/1/29.
//

#ifndef HISTOGRAMTRACKER_SEGMENT_H
#define HISTOGRAMTRACKER_SEGMENT_H

#include <opencv2/opencv.hpp>

struct Histogram
{
    double fR[255]{};
    double fG[255]{};
    double fB[255]{};

    double bR[255]{};
    double bG[255]{};
    double bB[255]{};
};

void ComputePosterior(const cv::Mat& img,const cv::Mat& mask,cv::Mat& fp,cv::Mat& bp);
void ComputeLvSet(const cv::Mat& mask,cv::Mat& lv_set);

#endif //HISTOGRAMTRACKER_SEGMENT_H
