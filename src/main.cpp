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

    int x_before = 240;
    int y_before = 200;
    int l_x_b = 159;
    int l_y_b = 130;

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

//    //here is some bp
//    BresenhamCircle(10,sample_points);
//    BresenhamCircle(Point{l_x_b,0},10,sample_points);
//    BresenhamCircle(Point{l_x_b,l_y_b},10,sample_points);
//    BresenhamCircle(Point{0,l_y_b},10,sample_points);

    vector<Point2d> lft,rgt;
    lft.emplace_back(0,0);
    lft.emplace_back(l_x_b,0);
    lft.emplace_back(l_x_b,l_y_b);
    lft.emplace_back(0,l_y_b);
    rgt.emplace_back(x_before,y_before);
    rgt.emplace_back(x_before + l_x_b,y_before);
    rgt.emplace_back(x_before + l_x_b,y_before + l_y_b);
    rgt.emplace_back(x_before,y_before + l_y_b);
    Mat H = findHomography(lft,rgt);
    H = H/H.at<double>(8);

    auto oppose = sse.log();
//    double n = 100;
//    for (int k = 0; k< n; ++k)
//    {
//        sample_points.emplace_back(0,l_y_b*k/n);
//        sample_points.emplace_back(l_x_b,l_y_b*k/n);
//        sample_points.emplace_back(l_x_b*k/n,0);
//        sample_points.emplace_back(l_x_b*k/n,l_y_b);
//    }
    cout<<sample_points.size()<<endl;
    for(int i=0;i<m.vertex_.size();i++){
        auto cost_function =
                new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1,6>(
                        new NumericDiffCostFunctor(m.vertex_[i],\
                        fposterior,\
                        bposterior,\
                        lv_set\
                        )
                ) ;

        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1), oppose.data());
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
    cv::perspectiveTransform(lft,rgt,H);
//    draw(result,rgt);
    m.Display(result,SE3d::exp(oppose));
    imshow("final",result);



    waitKey(0);
    return 0;
}