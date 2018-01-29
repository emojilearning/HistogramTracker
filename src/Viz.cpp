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

void draw(Mat& out1, std::vector<Point> p)
{
    line(out1,p[0],p[1],{0,255,0});
    line(out1,p[1],p[2],{0,255,0});
    line(out1,p[2],p[3],{0,255,0});
    line(out1,p[3],p[0],{0,255,0});
}
