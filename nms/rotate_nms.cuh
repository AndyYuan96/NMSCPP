#ifndef ROTATE_NMS_CUH
#define ROTATE_NMS_CUH
#include<vector>
#include <bitset>


extern "C"
void nms_rotated_cuda(const float *dets_host_ptr, float *dets_dev_ptr, std::vector<int>& ids, const int max_nms_input_nums, const int box_nums, const float iou_threshold, const int top_k);
extern "C"
void test_kernel();
#endif