//
// Created by flamming on 2018/1/30.
//

#include "Model.h"

#include <opencv2/core/eigen.hpp>

#include "Viz.h"

using namespace cv;
using namespace std;

Model::Model()
{
    double xl = -0.64;
    double yt = 0.48;
    double xr = 0.64;
    double yb = -0.48;
    vertex_.emplace_back(xl,yt,1);
    vertex_.emplace_back(xr,yt,1);
    vertex_.emplace_back(xr,yb,1);
    vertex_.emplace_back(xl,yb,1);
}

void Model::SetIntrinsic() {
    intrinsic_(0,0)=490.03599;    intrinsic_(0,1)=0;             intrinsic_(0,2)=329.14773;
    intrinsic_(1,0)=0;             intrinsic_(1,1)=490.06273;    intrinsic_(1,2)=240.90243;
    intrinsic_(2,0)=0;             intrinsic_(2,1)=0;             intrinsic_(2,2)=1;
}

void Model::Display(cv::Mat& canvas,Sophus::SE3d pose)
{
    Matx44d pose_cv;
    eigen2cv(pose.matrix(),pose_cv);
    vector<Point2d> vxv;
    for(auto v:vertex_)
    {
        Vec4d p = pose_cv * cv::Vec4d (v.x,v.y,v.z,1);
        p = p/p(3);
        Vec3d pt = Vec3d{p(0),p(1),p(2)};
        pt = pt/pt(2);

        pt = intrinsic_ * pt;
        vxv.emplace_back(pt(0),pt(1));
    }
    draw(canvas,vxv);
}

Point2d Model::Project2d(Point3d v,Sophus::SE3d pose)
{
    Matx44d pose_cv;
    eigen2cv(pose.matrix(),pose_cv);
    Vec4d p = pose_cv * cv::Vec4d (v.x,v.y,v.z,1);
    p = p/p(3);
    Vec3d pt = Vec3d{p(0),p(1),p(2)};
    pt = pt/pt(2);

    pt = intrinsic_ * pt;
    return {pt(0),pt(1)};
}
