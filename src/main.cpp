#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include "Viz.h"
#include "Segment.h"
#include "PoseEstimate.h"
#include "Region.h"
#include "Model.h"
#include "OcvYamlConfig.h"
#include "GlobalConfig.h"


using namespace cv;
using namespace std;
using namespace Sophus;



int main()
{
    OcvYamlConfig ocvYamlConfig("../config.yaml");
    Config::loadConfig(ocvYamlConfig);
    auto config = Config::configInstance();

    auto m = Model::GetInstance();
    m->SetIntrinsic();
    m->loadObj(config.objFile);

    //-1.98f, -2.90f, 37.47f, -40.90f, -207.77f, 27.48f
    Mat img = imread("/Users/flamming/WorkSpace/HistogramTracker/Files/Images/Red.png");
    imshow("result",img);


    Matrix4d pv;
    Vector3d last_translation = {-2.58,2.90,37};
    Eigen::Quaterniond last_rot = {0.13,-0.3,-0.35,1};//{0.13,0.85,0.39,-0.30};
    SE3d sse(SO3d(last_rot),last_translation);



    Mat out0 = img.clone();
    m->displayCV(sse,{0,255,0},out0);
    imshow("result1",out0);
    waitKey(0);

    Mat mask = Mat::zeros(img.size(),CV_8U);
    //mask ;//= m->DrawMask(img,sse);
    Mat temp = imread("/Users/flamming/WorkSpace/HistogramTracker/Files/Masks/Red_Mask.png");

//    for (int k = 0; k < temp.total(); ++k) {
//        mask.at<unsigned char>(k) =  temp.at<int>(k);
//    }

    cvtColor(temp, mask, COLOR_BGR2GRAY);

    cv::Mat fposterior,bposterior;
    ComputePosterior(img,mask,fposterior,bposterior);
    (Mat(mask/255)).convertTo(fposterior,CV_64F);
    (Mat((~mask)/255)).convertTo(bposterior,CV_64F);
    Mat foreth = (fposterior>bposterior)*255;
    imshow("post",foreth);

    Mat lv_set;
    ComputeLvSet(mask ,lv_set);

    ceres::Problem problem;

    vector<Point> sample_points;// = {{0,0},{l_x_b,0},{0,l_y_b},{l_x_b,l_y_b}};

    BresenhamCircle(10,sample_points);

    auto rot = sse.so3().log();
    auto& t = sse.translation();
    double pose_op[6] = {rot(0),rot(1),rot(2),t(0),t(1),t(2)};
    double* pose_ptr = pose_op;
    Mat vext = m->vertices_hom_;
    cout<<vext.cols<<endl;

    for(int i=0;i<m->vertex_.size();i++){
        for (int j = 0; j < sample_points.size(); ++j) {
            auto cost_function = new CostFunctionByJac(m->vertex_[i],\
                        fposterior,
                        bposterior,
                        lv_set, sample_points[j]) ;

            problem.AddResidualBlock(cost_function, 0, pose_ptr);//new ceres::CauchyLoss(1)
        }

    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.initial_trust_region_radius = 1e-8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    imshow("lv_set",lv_set);

    Mat result = img.clone();
//    draw(result,rgt);

    sse = SE3d(SO3d::exp(Eigen::Vector3d(pose_op[0],pose_op[1],pose_op[2])),
            Eigen::Vector3d(pose_op[3],pose_op[4],pose_op[5]));
    SE3d Tp(sse);
    m->displayCV(sse,{255,0,0},result);
    imshow("final",result);

    waitKey(0);
    return 0;
}