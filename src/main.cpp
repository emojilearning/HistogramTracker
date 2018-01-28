#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

using namespace cv;
using namespace std;

void draw(Mat& out1, int x_before,int y_before,int l_x_b,int l_y_b)
{
    line(out1,{x_before,y_before},{x_before+l_x_b,y_before},{0,255,0});
    line(out1,{x_before,y_before},{x_before,l_y_b + y_before},{0,255,0});
    line(out1,{x_before+l_x_b,y_before},{x_before+l_x_b,y_before+l_y_b},{0,255,0});
    line(out1,{x_before,y_before+l_y_b},{x_before+l_x_b,y_before+l_y_b},{0,255,0});
}

struct Histogram
{
    double fR[255]{};
    double fG[255]{};
    double fB[255]{};

    double bR[255]{};
    double bG[255]{};
    double bB[255]{};
};


struct NumericDiffCostFunctor {

    NumericDiffCostFunctor(cv::Point &X, cv::Mat &fwd, cv::Mat &bg, cv::Mat &dt_map) :
            X_(X), fwd_(fwd), bg_(bg), dt_map_(dt_map) {}

    bool operator()(const double *pose, double *residual) const {
        double E = 0;
        double x = pose[0] + X_.x;
        double y = pose[1] + X_.y;
        int xf = (int)floor(x);
        int yf = (int)floor(y);
        int xc = (int)ceil(x);
        int yc = (int)ceil(y);
        auto lb = dt_map_.at<float>(cv::Point(xf,yf));
        auto rb = dt_map_.at<float>(cv::Point(xc,yf));
        auto lt = dt_map_.at<float>(cv::Point(xf,yc));
        auto rt = dt_map_.at<float>(cv::Point(xc,yc));
        auto mb = (lb + (x - xf)*(rb - lb));
        auto mt = (lt + (x - xf)*(rt - lt));

        auto dt = mb + (mt - mb)*(y - yf);
        double Theta = (dt);
//        auto He = (1.0) / ((1) + ceres::exp(Theta));
//        std::cout<<lb<<std::endl;

//        E = He;
        residual[0] = Theta;
        return true;
    }


    const cv::Point X_;
    const cv::Mat dt_map_;
    const cv::Mat fwd_;
    const cv::Mat bg_;
};

int main()
{
    Mat img = imread("../origin.png");
    imshow("result",img);

    int x_before = 260;
    int y_before = 134;
    int l_x_b = 139;
    int l_y_b = 111;

    double initial_pose[2] = {(double)x_before,(double)y_before};

    Mat out1 = img.clone();
    draw(out1,x_before,y_before,l_x_b,l_y_b);
    imshow("result1",out1);

    Mat mask = Mat::zeros(img.rows,img.cols,CV_8U);

    Rect mm(x_before,y_before,l_x_b,l_y_b);
    mask(mm).setTo(255);
//    imshow("mask",mask);

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
//        std::cout<<fposterior.at<double>(j)<<" "<<bposterior.at<double>(j)<<std::endl;
    }

    Mat foreth = (fposterior>bposterior)*255;
    imshow("post",foreth);

    Mat dt_map,dt_mapr;
    distanceTransform(foreth,dt_map,DIST_L2,3);
    distanceTransform(~foreth,dt_mapr,DIST_L2,3);
    dt_map = dt_map - dt_mapr;


    ceres::Problem problem;

    Point cornor[] = {{0,0},{l_x_b,0},{0,l_y_b},{l_x_b,l_y_b}};
    for(int i=0;i<4;i++){
        auto cost_function =
                new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1,2>(
                        new NumericDiffCostFunctor(cornor[i],\
                        fposterior,\
                        bposterior,\
                        dt_map\
                        )
                ) ;
//        cout<<dt_map.at<float>(y_before,x_before)<<endl;
//        cout<<dt_map.at<float>(y_before,x_before + l_x_b)<<endl;
//        cout<<dt_map.at<float>(y_before + l_y_b,x_before)<<endl;
//        cout<<dt_map.at<float>(y_before + l_y_b,x_before + l_x_b)<<endl;

        problem.AddResidualBlock(cost_function, NULL, initial_pose);
    }

    ceres::Solver::Options options;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << initial_pose[0] << "\n";
    std::cout << initial_pose[1] << "\n";

    imshow("dt_map",dt_map);

    Mat result = img.clone();
    draw(result,initial_pose[0],initial_pose[1],l_x_b,l_y_b);
    imshow("final",result);



    waitKey(0);
    return 0;
}