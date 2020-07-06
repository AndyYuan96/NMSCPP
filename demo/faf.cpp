#include "faf.h"

FAF::FAF(const std::string& model_path, PixorParams& feature_params, const int depth, const std::string& Root, const std::string& source, int frame_load, int sequence_start, int sequence_end, float *anchor_x, float *anchor_y)
{
    int x = (feature_params.x_max - feature_params.x_min) / feature_params.x_division;
    int y = (feature_params.y_max - feature_params.y_min) / feature_params.y_division;
    int z = (feature_params.z_max - feature_params.z_min) / feature_params.z_division;

    range_x_ = int(feature_params.x_max - feature_params.x_min);
    range_y_ = int(feature_params.y_max - feature_params.y_min);

    max_nms_input_anchor_nums_ = 200;

    feature_size_x_ = 75;
    feature_size_y_ = 75;
    anchor_nums_ = feature_size_x_ * feature_size_y_ * 4;
    feature_params_ = feature_params;
    feature_.reset(new float[depth * z * x * y]);
    output_cls_.reset(new float[anchor_nums_ * 1]);
    output_reg_.reset(new float[anchor_nums_ * 10]);

    decoded_output_ = new float* [max_nms_input_anchor_nums_];
    for(int i = 0; i < max_nms_input_anchor_nums_; i++)
    {
        decoded_output_[i] = new float[10];
    }

    continuous_sort_output_.reset(new float[max_nms_input_anchor_nums_*5]);

    thres_anchor_ids_.resize(max_nms_input_anchor_nums_);

    anchor_x_dims_[0] = (2.0/range_x_);
    anchor_x_dims_[1] = (4.0/range_x_);
    anchor_x_dims_[2] = (4.0/range_x_);
    anchor_x_dims_[3] = (8.0/range_x_);

    anchor_y_dims_[0] = (4.0/range_y_);
    anchor_y_dims_[1] = (2.0/range_y_);
    anchor_y_dims_[2] = (8.0/range_y_);
    anchor_y_dims_[3] = (4.0/range_y_);

    anchor_positions_ = new float[anchor_nums_][2];
    int index = 0;
    for(int i = 0; i < feature_size_y_; i++)
        for(int j = 0; j < feature_size_x_; j++)
        {
            anchor_positions_[index][0] = float(j + 0.5) / feature_size_x_;
            anchor_positions_[index][1] = float(i + 0.5) / feature_size_y_;
            ++index;
        }

    infer_output_.resize(2);
    infer_output_[1] = output_cls_.get();
    infer_output_[0] = output_reg_.get();

    std::cout << Root << std::endl << source << std::endl;
    std::cout << model_path << std::endl;
    std::cout << "feature size: " << std::endl;
    std::cout << "x  y  z" << std::endl;
    std::cout << x << " " << y << " " << z << std::endl;

    loader_.reset(new MultiFramePixorFeatureAnchorLabelPreprocess(Root,source,feature_params,frame_load,sequence_start,sequence_end));
    inference_ptr_.reset(new TRTInference(model_path, x,y, false, 30,"faf"));
    std::cout << "finished init " << std::endl;
}

FAF::~FAF()
{
    for(int i = 0; i < max_nms_input_anchor_nums_; i++)
    {
        delete []decoded_output_[i];
    }
    delete []decoded_output_;

    delete []anchor_positions_;
}

void FAF::load(int index, pcl::PointCloud<pcl::PointXYZI>::Ptr display_cloud)
{
    
    loader_->generateFeature(feature_.get(),index,display_cloud);
}

void FAF::inference(std::vector<Bbox3DWithPrediction>& detection_results, const float iou_thres, const float score_thres, const int top_k)
{
    clock_t start, finish;        
    start = clock();

    inference_ptr_->doInference(feature_.get(),infer_output_);

    auto output_reg = output_reg_.get();
    auto output_cls = output_cls_.get();

    int thres_anchor_counts = 0;

    for(int i = 0; i < anchor_nums_; i++)
    {
        if(output_cls[i] >= score_thres)
        {
            thres_anchor_ids_[thres_anchor_counts] = i;
            ++thres_anchor_counts;
        }
    }

    if(thres_anchor_counts > max_nms_input_anchor_nums_)
    {
        std::cout << "ERROR >> thres output anchor's num > max_nms_input_anchor_nums_" << std::endl;
        exit(EXIT_FAILURE);
    }

    // deocde output according to thres_output_number
    decode_anchor(thres_anchor_counts);
    std::sort(decoded_output_,decoded_output_ + thres_anchor_counts,cmp);

    // convert to continuous output for nms, only save w h x y theta
    auto continuous_output = continuous_sort_output_.get();
    for(int i = 0; i < thres_anchor_counts; ++i)
    {
        for(int j = 0; j < 5; ++j)
        {
            continuous_output[i * 5 + j] = decoded_output_[i][j];
        }
    }

    std::vector<int> nms_ids;
    nms_rotated_cuda(continuous_output,cuda_detection_ptr.get(),nms_ids,max_nms_input_anchor_nums_,thres_anchor_counts,iou_thres,top_k);

    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "inference time : " << duration << std::endl;

    // for(int i = 0; i < nms_ids.size(); i++)
    // {
    //     for(int j = 0; j < 5; j++)
    //     {
    //         std::cout << decoded_output_[i][j] << " " ; // << continuous_output[nms_ids[i] * 5 + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<Bbox3DWithPrediction> empty_vt;
    detection_results.swap(empty_vt);

    decode_output(nms_ids, detection_results);

}

void FAF::decode_output(std::vector<int>& nms_ids, std::vector<Bbox3DWithPrediction>& output)
{
    for(int i = 0; i < nms_ids.size(); ++i)
    {
        int cur_id = nms_ids[i];

        // decode to car coordinate, w h x y theta p_x,p_y,p2_x,p2_y
        Bbox3DWithPrediction tmp_box;

        tmp_box.bx =  decoded_output_[cur_id][1];
        tmp_box.by = decoded_output_[cur_id][0];
        tmp_box.bz = 1.5;
        tmp_box.tx = decoded_output_[cur_id][3];
        tmp_box.ty = decoded_output_[cur_id][2];
        tmp_box.tz = -1.0;

        tmp_box.rz = decoded_output_[cur_id][4];

        tmp_box.predict_x.resize(2);
        tmp_box.predict_y.resize(2);

        tmp_box.predict_x[0] = decoded_output_[cur_id][6] * 120 - 60;
        tmp_box.predict_x[1] = decoded_output_[cur_id][8] * 120 - 60;

        tmp_box.predict_y[0] = decoded_output_[cur_id][5] * 120 - 60;
        tmp_box.predict_y[1] = decoded_output_[cur_id][7] * 120 - 60;

        tmp_box.score = decoded_output_[cur_id][9];

        output.push_back(tmp_box);
    }
}

void FAF::decode_anchor(int thres_anchor_counts)
{
    int anchor_wh = 0;
    int anchor_position = 0;

    auto output_reg = output_reg_.get();
    auto output_cls = output_cls_.get();

    int cur_id = 0;
    int cur_reg_position = 0;

    for(int i = 0; i < thres_anchor_counts; i++)
    {
        cur_id = thres_anchor_ids_[i];
        cur_reg_position = cur_id * 10;

        anchor_wh = cur_id % 4;
        anchor_position = cur_id / 4;

        decoded_output_[i][0] = std::exp(output_reg[cur_reg_position + 0]) * anchor_x_dims_[anchor_wh]* 120;
        decoded_output_[i][1] = std::exp(output_reg[cur_reg_position + 1]) * anchor_y_dims_[anchor_wh]* 120;

        decoded_output_[i][2] = (output_reg[cur_reg_position + 2] * anchor_x_dims_[anchor_wh] + anchor_positions_[anchor_position][0]) * 120 - 60;
        decoded_output_[i][3] = (output_reg[cur_reg_position + 3] * anchor_y_dims_[anchor_wh] + anchor_positions_[anchor_position][1]) * 120 - 60;

        decoded_output_[i][4] = std::atan2(output_reg[cur_reg_position + 5],output_reg[cur_reg_position + 4]) * -0.5;  //convert to shun shizhen
        decoded_output_[i][5] = (output_reg[cur_reg_position + 6] * anchor_x_dims_[anchor_wh] + anchor_positions_[anchor_position][0]);
        decoded_output_[i][6] = (output_reg[cur_reg_position + 7] * anchor_y_dims_[anchor_wh] + anchor_positions_[anchor_position][1]);
        decoded_output_[i][7] = (output_reg[cur_reg_position + 8] * anchor_x_dims_[anchor_wh] + anchor_positions_[anchor_position][0]);
        decoded_output_[i][8] = (output_reg[cur_reg_position + 9] * anchor_y_dims_[anchor_wh] + anchor_positions_[anchor_position][1]);

        decoded_output_[i][9] =  output_cls[cur_id];
    }
}

int FAF::getDatasetSize()
{
    return loader_->getDatasetSize();
}