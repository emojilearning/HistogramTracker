//
// Created by flamming on 2018/1/30.
//
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "glm/glm.h"

#ifndef HISTOGRAMTRACKER_MODEL_H
#define HISTOGRAMTRACKER_MODEL_H

class Model
{
public:
    Model();
    Model(std::string model_file);

    static Model* GetInstance(){
        static Model* m = new Model;
        return m;
    }
    void SetIntrinsic();
    void Display(cv::Mat& canvas,Sophus::SE3d pose);
    cv::Mat DrawMask(cv::Mat& mask,Sophus::SE3d pose);
    cv::Point2d Project2d(cv::Point3d vertex,Sophus::SE3d pose);
    void loadObj(const std::string& filename);

    //cv part
    void getContourPointsAndIts3DPoints( Sophus::SE3d &pose,std::vector<cv::Point3d> &verticesContour_Xs,
                                         std::vector<cv::Point2d> &verticesContour_xs,std::vector<cv::Point> &resContour);

    void displayCV( Sophus::SE3d &pose,const cv::Scalar &color, cv::Mat& frame);

    void getVisualableVertices( Sophus::SE3d& pose, cv::Mat& vis_vertices);
    void project3D_2D( Sophus::SE3d &pose, const cv::Mat& visible_Xs,  cv::Mat &visible_xs);
    bool pointInFrame(const cv::Point &p);
    cv::Point X_to_x(const cv::Point3f &X,const cv::Mat &extrisic);
    int VerticesCount() {return model_->numvertices;}

    GLMmodel* model_;
    cv::Mat vertices_hom_;

    cv::Matx33d intrinsic_;
    cv::Matx34d intrinsic34;
    std::vector<cv::Point3d> vertex_;

    inline void Cross(double* u, double* v, double* n) {
        n[0] = u[1] * v[2] - u[2] * v[1];
        n[1] = u[2] * v[0] - u[0] * v[2];
        n[2] = u[0] * v[1] - u[1] * v[0];
    }

    inline void Normalize(double* v) {
        float l = (float)sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] /= l, v[1] /= l, v[2] /= l;
    }
};

#endif //HISTOGRAMTRACKER_MODEL_H
