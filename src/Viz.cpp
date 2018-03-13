//
// Created by flamming on 2018/1/29.
//

#include "Viz.h"

using namespace cv;

void draw(Mat& out1, int x_before,int y_before,int l_x_b,int l_y_b)
{
    line(out1,{x_before,y_before},{x_before+l_x_b,y_before},{0,255,0});
    line(out1,{x_before,y_before},{x_before,l_y_b + y_before},{0,255,0});
    line(out1,{x_before+l_x_b,y_before},{x_before+l_x_b,y_before+l_y_b},{0,255,0});
    line(out1,{x_before,y_before+l_y_b},{x_before+l_x_b,y_before+l_y_b},{0,255,0});
}

void draw(Mat& out1, std::vector<Point2d> p,cv::Scalar s)
{
    for (int i = 0; i < p.size() ; ++i) {
        line(out1,p[i],p[(i+1)%p.size()],s);
    }
}
