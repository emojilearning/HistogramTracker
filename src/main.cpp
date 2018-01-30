#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include "Viz.h"
#include "Segment.h"
#include "PoseEstimate.h"
#include "Region.h"


using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("../origin.png");
    imshow("result",img);

    int x_before = 265;
    int y_before = 135;
    int l_x_b = 139;
    int l_y_b = 111;

    double initial_pose[2] = {(double)x_before,(double)y_before};


    Mat out1 = img.clone();
    draw(out1,x_before,y_before,l_x_b,l_y_b);
    imshow("result1",out1);

    Mat mask = Mat::zeros(img.rows,img.cols,CV_8U);
    Rect mm(x_before,y_before,l_x_b,l_y_b);
    mask(mm).setTo(255);
    imshow("mask",mask);


    cv::Mat fposterior,bposterior;
    ComputePosterior(img,mask,fposterior,bposterior);
    Mat foreth = (fposterior>bposterior)*255;
    imshow("post",foreth);

    Mat lv_set;
    ComputeLvSet(mask,lv_set);

    ceres::Problem problem;

    vector<Point> sample_points;// = {{0,0},{l_x_b,0},{0,l_y_b},{l_x_b,l_y_b}};

    vector<Point> temp;
    BresenhamCircle(10,temp);
    sample_points.insert(sample_points.end(), temp.begin(),temp.end());
    BresenhamCircle(Point{l_x_b,0},10,temp);
    sample_points.insert(sample_points.end(), temp.begin(),temp.end());
    BresenhamCircle(Point{l_x_b,l_y_b},10,temp);
    sample_points.insert(sample_points.end(), temp.begin(),temp.end());
    BresenhamCircle(Point{0,l_y_b},10,temp);
    sample_points.insert(sample_points.end(), temp.begin(),temp.end());

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


//    for (int k = 0; k< 10; ++k)
//    {
//        sample_points.emplace_back(0,l_y_b*k/10.0);
//        sample_points.emplace_back(l_x_b,l_y_b*k/10.0);
//        sample_points.emplace_back(l_x_b*k/10.0,0);
//        sample_points.emplace_back(l_x_b*k/10.0,l_y_b);
//    }
    for(int i=0;i<sample_points.size();i++){
        auto cost_function =
                new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1,9>(
                        new NumericDiffCostFunctor(sample_points[i],\
                        fposterior,\
                        bposterior,\
                        lv_set\
                        )
                ) ;

        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1), H.ptr<double>(0));
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << initial_pose[0] << "\n";
    std::cout << initial_pose[1] << "\n";

    imshow("lv_set",lv_set);

    Mat result = img.clone();
    cv::perspectiveTransform(lft,rgt,H);
    draw(result,rgt);
    imshow("final",result);



    waitKey(0);
    return 0;
}