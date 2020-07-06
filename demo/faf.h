#ifndef FAF_H
#define FAF_H
#include "preprocessor/preprocess/multi_frame_pixor_feature_anchor_label_preprocess.h"
#include "preprocessor/util/util.h"
#include "../rs_cnn_base/include/rs_perception/cnn_detection/rs_cnn_base/trt_infer.h"
#include <string>
#include <algorithm>
#include <memory>
#include <cmath>


extern "C"
void nms_rotated_cuda(const float *dets_host_ptr, float *dets_dev_ptr, std::vector<int>& ids, const int max_nms_input_nums, const int box_nums, const float iou_threshold, const int top_k);

using namespace robosense::perception;

bool cmp(float *c1, float * c2)
{
    return c1[9] > c2[9];
}

// new dynamic array, only 1 dimension, if 2 dimension array, second dimension must be const value.
class FAF
{
public:
    FAF(const std::string& model_path, PixorParams& feature_params, const int depth, const std::string& Root, const std::string& source, int frame_load, int sequence_start, int sequence_end, float *anchor_x, float *anchor_y);
    ~FAF();
    void load(int index, pcl::PointCloud<pcl::PointXYZI>::Ptr display_cloud);

    void inference(std::vector<Bbox3DWithPrediction>& detection_results, const float iou_thres = 0.1, const float score_thres = 0.1, const int top_k = 30);

    void decode_anchor(int counts);

    void decode_output(std::vector<int>& nms_ids, std::vector<Bbox3DWithPrediction>& output);

    int getDatasetSize();

    float (*anchor_positions_)[2];

    std::shared_ptr<TRTInference> inference_ptr_;
    std::shared_ptr<float> feature_;
    std::shared_ptr<float> output_reg_;
    std::shared_ptr<float> output_cls_;
    float* (*decoded_output_);

    std::shared_ptr<float> continuous_sort_output_;

    std::shared_ptr<float> cuda_detection_ptr;

    std::vector<int> thres_anchor_ids_;

    std::vector<float*> infer_output_;
    float anchor_x_dims_[4];
    float anchor_y_dims_[4];
    int feature_size_x_;
    int feature_size_y_;
    float range_x_;
    float range_y_;
    int anchor_nums_;
    int max_nms_input_anchor_nums_;
    PixorParams feature_params_;
    std::shared_ptr<MultiFramePixorFeatureAnchorLabelPreprocess> loader_;    

};
#endif