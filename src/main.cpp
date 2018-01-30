#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include "Viz.h"
#include "Segment.h"
#include "PoseEstimate.h"
#include "Region.h"
#include "Model.h"


using namespace cv;
using namespace std;
using namespace Sophus;

int main()
{
    Mat img = imread("../origin.png");
    imshow("result",img);

    int x_before = 248;
    int y_before = 212;
    int l_x_b = 161;
    int l_y_b = 120;

    double initial_pose[2] = {(double)x_before,(double)y_before};

    Model m;
    m.SetIntrinsic();
    Matrix4d pv;
    pv<<    1,0,0,0,
            0,1,0,0.25,
            0,0,1,2.9,
            0,0,0,1;
    SE3d sse(pv);

    Mat out0 = img.clone();
    m.Display(out0,sse);
    imshow("result1",out0);

    Mat mask = Mat::zeros(img.rows,img.cols,CV_8U);
    Rect mm(x_before,y_before,l_x_b,l_y_b);
    mask(mm).setTo(255);


    cv::Mat fposterior,bposterior;
    ComputePosterior(img,mask,fposterior,bposterior);
    Mat foreth = (fposterior>bposterior)*255;
    imshow("post",foreth);

    Mat lv_set;
    ComputeLvSet(mask,lv_set);

    ceres::Problem problem;

    vector<Point> sample_points;// = {{0,0},{l_x_b,0},{0,l_y_b},{l_x_b,l_y_b}};

    auto rot = sse.so3().log();
    auto& t = sse.translation();
    double pose_op[6] = {rot(0),rot(1),rot(2),t(0),t(1),t(2)};

    for(int i=0;i<m.vertex_.size();i++){
        auto cost_function =
                new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1,6>(
                        new NumericDiffCostFunctor(m.vertex_[i],\
                        fposterior,\
                        bposterior,\
                        lv_set\
                        )
                ) ;

        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1), pose_op);
    }

    ceres::Solver::Options options;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << initial_pose[0] << "\n";
    std::cout << initial_pose[1] << "\n";

    imshow("lv_set",lv_set);

    Mat result = img.clone();
//    draw(result,rgt);

    SE3d Tp(SO3d::exp(Eigen::Vector3d(pose_op[0],pose_op[1],pose_op[2])),
         Eigen::Vector3d(pose_op[3],pose_op[4],pose_op[5]));
    m.Display(result,Tp);
    imshow("final",result);



    waitKey(0);
    return 0;
}