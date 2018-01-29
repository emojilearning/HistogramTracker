//
// Created by flamming on 2018/1/29.
//

#include "Segment.h"

using namespace cv;

void ComputePosterior(const cv::Mat& img,const cv::Mat& mask,cv::Mat& fp,cv::Mat& bp)
{

    Histogram cur_histogram;
    double fnum = 0;
    double gnum = 0;
    for (int i = 0; i < mask.size().area(); ++i) {
        auto BGR = img.at<Vec3b>(i);
        if(mask.at<unsigned char>(i))
        {
            cur_histogram.fB[BGR[0]]++;
            cur_histogram.fG[BGR[1]]++;
            cur_histogram.fR[BGR[2]]++;
            fnum++;
        }
        else
        {
            cur_histogram.bB[BGR[0]]++;
            cur_histogram.bG[BGR[1]]++;
            cur_histogram.bR[BGR[2]]++;
            gnum++;
        }
    }

    cv::Mat fposterior = Mat::zeros(img.size(),CV_64F);
    cv::Mat bposterior = Mat::zeros(img.size(),CV_64F);
    for (int j = 0; j < img.size().area(); ++j) {
        auto p = img.at<Vec3b>(j);
        auto flikelihood = (cur_histogram.fB[p[0]]/fnum
                            * cur_histogram.fG[p[1]]/fnum
                            *cur_histogram.fR[p[2]]/fnum);
        auto blikelihood = (cur_histogram.bB[p[0]]/gnum
                            * cur_histogram.bG[p[1]]/gnum
                            * cur_histogram.bR[p[2]]/gnum);
        fposterior.at<double>(j) = flikelihood * (fnum/(fnum + gnum))*fnum/(fnum + gnum)*fnum/(fnum + gnum);
        bposterior.at<double>(j) = blikelihood * (gnum/(fnum + gnum))*gnum/(fnum + gnum)*gnum/(fnum + gnum);
        if(fposterior.at<double>(j)&&bposterior.at<double>(j))
        {
            auto s = fposterior.at<double>(j)+bposterior.at<double>(j);
            fposterior.at<double>(j) = fposterior.at<double>(j)/s;
            bposterior.at<double>(j) = bposterior.at<double>(j)/s;
        }
//        std::cout<<fposterior.at<double>(j)<<" "<<bposterior.at<double>(j)<<std::endl;
    }
    fp = fposterior;
    bp = bposterior;
}

//Be cautious that lvset is opposite to distance map
void ComputeLvSet(const cv::Mat& mask,cv::Mat& lv_set)
{
    Mat dt_mapr;

    //initial as mask or foreth?(which is not so accurate a map but faster if valid
    distanceTransform(mask,lv_set,DIST_L2,3);
    distanceTransform(~mask,dt_mapr,DIST_L2,3);
    //Φ(x) = −d(x),∀x ∈ Ωf and Φ(x) = d(x),∀x ∈ Ωb.
    lv_set = -lv_set + dt_mapr;
}

