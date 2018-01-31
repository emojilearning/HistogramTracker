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
    pv<<    1,0,0,-1.9,
            0,1,0,2.9,
            0,0,1,37.0,
            0,0,0,1;
    Vector3d last_translation = {-1.98,-2.90,37};
    SE3d sse(pv);



    Mat out0 = img.clone();
    m->displayCV(sse,{0,255,0},out0);
    imshow("result1",out0);
    waitKey(0);

     Mat mask;
    mask = m->DrawMask(img,sse);


    cv::Mat fposterior,bposterior;
    ComputePosterior(img,mask,fposterior,bposterior);
    Mat foreth = (fposterior>bposterior)*255;
    imshow("post",foreth);

    Mat lv_set;
    ComputeLvSet(foreth ,lv_set);

    ceres::Problem problem;

    vector<Point> sample_points;// = {{0,0},{l_x_b,0},{0,l_y_b},{l_x_b,l_y_b}};

    BresenhamCircle(10,sample_points);

    auto rot = sse.so3().log();
    auto& t = sse.translation();
    double pose_op[6] = {rot(0),rot(1),rot(2),t(0),t(1),t(2)};

    for(int i=0;i<m->vertex_.size();i++){
        for (int j = 0; j < sample_points.size(); ++j) {
            auto cost_function =
                    new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1,6>(
                            new NumericDiffCostFunctor(m->vertex_[i],sample_points[j],\
                        fposterior,\
                        bposterior,\
                        lv_set\
                        )
                    ) ;

            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1), pose_op);
        }

    }

    ceres::Solver::Options options;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    imshow("lv_set",lv_set);

    Mat result = img.clone();
//    draw(result,rgt);

    SE3d Tp(SO3d::exp(Eigen::Vector3d(pose_op[0],pose_op[1],pose_op[2])),
         Eigen::Vector3d(pose_op[3],pose_op[4],pose_op[5]));
    m->displayCV(Tp,{0,255,0},result);
    imshow("final",result);

    waitKey(0);
    return 0;
}