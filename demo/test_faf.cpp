#include "faf.h"
#include "preprocessor/util/util.h"

#include<iostream>
#include<string>

using namespace robosense::perception;

const std::string model_path = "/media/ovo/file1/CWD/FaF_inference/src/faf/model/encrypt";
const std::string Root = "/media/ovo/file1/dataset/waymo_tracking_1hz";
const std::string source = "/media/ovo/file1/dataset/waymo_tracking/val";

const int frame_load = 5;
const int depth = 3;
float anchor_x[4] = {0,0,0,0};
float anchor_y[4] = {0,0,0,0};

int sequence_start = 0;
int sequence_end = 0;

pcl::PointCloud<pcl::PointXYZI>::Ptr display_cloud;



int main()
{
    PixorParams feature_params;
    feature_params.x_min = -60.0;
    feature_params.x_max = 60.0;
    feature_params.y_min = -60.0;
    feature_params.y_max = 60.0;
    feature_params.z_min = -1.6;
    feature_params.z_max = 4;
    feature_params.x_division = 0.2;
    feature_params.y_division = 0.2;
    feature_params.z_division = 0.2;
 
    FAF * model_ptr = new FAF(model_path,feature_params,depth,Root,source,frame_load,sequence_start,sequence_end,anchor_x,anchor_y);
    int dataset_size = model_ptr->getDatasetSize();

    std::vector<Bbox3DWithPrediction> detection_results;
    // test_kernel();

    for(int i = 0; i < 10; i++)
    {
        model_ptr->load(i,display_cloud);
        model_ptr->inference(detection_results);
        
        // for(int j = 0; j < detection_results.size(); ++j)
        // {
        //     for(int k = 0; k < detection_results[j].size(); ++k)
        //     {
        //         std::cout << detection_results[j][k] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }


}