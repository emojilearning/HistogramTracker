//
// Created by flamming on 2018/1/30.
//

#include "Model.h"

#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>

#include "Viz.h"
#include "GlobalConfig.h"


using namespace cv;
using namespace std;

Model::Model()
{
    model_ = NULL;
}

Model::Model(std::string model_file)
{
    model_ = NULL;
    this->loadObj(model_file);
}


void Model::SetIntrinsic() {
    auto config = Config::configInstance();
    intrinsic_(0,0)=config.FX;    intrinsic_(0,1)=0;             intrinsic_(0,2)=config.CX;
    intrinsic_(1,0)=0;             intrinsic_(1,1)=config.FY;    intrinsic_(1,2)=config.CY;
    intrinsic_(2,0)=0;             intrinsic_(2,1)=0;             intrinsic_(2,2)=1;

    intrinsic34(0,0)=config.FX;    intrinsic34(0,1)=0;             intrinsic34(0,2)=config.CX;      intrinsic34(0,3)=0;
    intrinsic34(1,0)=0;             intrinsic34(1,1)=config.FY;    intrinsic34(1,2)=config.CY;      intrinsic34(1,3)=0;
    intrinsic34(2,0)=0;             intrinsic34(2,1)=0;             intrinsic34(2,2)=1;             intrinsic34(2,3)=0;
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
        cout<<pt<<endl;
    }
    draw(canvas,vxv);
}

Mat Model::DrawMask(cv::Mat& img,Sophus::SE3d pose)
{
    Mat temp(Mat::zeros(img.size(),CV_8U));
    displayCV(pose,{255,255,255},temp);
    vector<vector<Point>> contours;
    findContours(temp,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    drawContours(temp,contours, -1, CV_RGB(255, 255, 255), CV_FILLED);
    return temp;
}

void Model::loadObj(const std::string& filename) {

    if (model_)
        glmDelete(model_);

    model_ = glmReadOBJ(const_cast<char*>(filename.c_str()));
    CHECK(model_) << "failed to load model";
    vertex_.clear();
    vertices_hom_ = cv::Mat::zeros(4, model_->numvertices, CV_64F);
    for (int i = 0; i <= model_->numvertices; ++i) {
        vertex_.emplace_back(model_->vertices[3 * (i)+0],
                             model_->vertices[3 * (i)+1],
                             model_->vertices[3 * (i)+2]);
        vertices_hom_.at<double>(0, i) = model_->vertices[3 * (i)+0];
        vertices_hom_.at<double>(1, i) = model_->vertices[3 * (i)+1];
        vertices_hom_.at<double>(2, i) = model_->vertices[3 * (i)+2];
        vertices_hom_.at<double>(3, i) = 1;
    }
    SetIntrinsic();
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


