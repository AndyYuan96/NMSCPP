#include "rotate_nms.cuh"
#include "rotate_nms_utils.h"
#include <vector>
#include <iostream>

#define DIVUP(m,n) ((m + n - 1)/n)
int64_t const threadsPerBlock = sizeof(unsigned long long) * 8;

__global__ void testKernel(float* val)
{
   int idx = threadIdx.x;
   val[idx] += 1;
   printf("current idx: %d", idx);
}

__device__ inline float devIoU(float const * const a, float const * const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
  }


template <typename T>
__global__ void nms_rotated_cuda_kernel(
    const T* dev_boxes,
    const int n_boxes,
    const float iou_threshold,
    unsigned long long *mask_dev_ptr)
{
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
    
    __shared__ T block_boxes[threadsPerBlock*5];

    if(threadIdx.x < col_size)
    {
        block_boxes[threadIdx.x * 5 + 0] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
        block_boxes[(threadIdx.x * 5) + 2] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
    }

    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const T* cur_box = dev_boxes + cur_box_idx * 5;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) 
        {
            start = threadIdx.x + 1;
        }

        for (i = start; i < col_size; i++) 
        {
            if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5) > iou_threshold)  //
            {
                t |= 1ULL << i;
            }
        }

        const int col_blocks = DIVUP(n_boxes,threadsPerBlock);
        mask_dev_ptr[cur_box_idx * col_blocks + col_start] = t;

        if(row_start == 0 && col_start == 0)
        {
            if(threadIdx.x == 1)
            {
                // T iou = single_box_iou_rotated<T>(cur_box, block_boxes + 11 * 5);
                // printf("%f %f %f %f %f \n",cur_box[0],cur_box[1],cur_box[2],cur_box[3],cur_box[4]); //  
                // printf("%f %f %f %f %f \n",*(block_boxes + 11 * 5 + 0),*(block_boxes + 11 * 5 + 1),*(block_boxes + 11 * 5 + 2),*(block_boxes + 11 * 5 + 3),*(block_boxes + 11 * 5 + 4));
                // printf(" %llu\n",t);
                // printf(" %llu\n",mask_dev_ptr[cur_box_idx * col_blocks + col_start]);
                // printf(" %d\n",cur_box_idx * col_blocks + col_start);
                // printf("iou %f\n",iou);
            }
        }
    }
}

__global__ void print_mask(unsigned long long* mask, int box_nums, int cols)
{
    int idx = threadIdx.x;
    printf("enter print mask");

    if(idx < box_nums)
    {
        for(int i = 0; i < cols; i++)
        {

        }
    }
}

void test_kernel()
{
    printf("enter kernel\n");

    float test_host[5] = {0};
    
    float *test_dev = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&test_dev, 5 * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(test_dev, test_host, 5 * sizeof(float),cudaMemcpyHostToDevice));

    testKernel<<<1,5>>>(test_dev);
    
    HANDLE_ERROR(cudaMemcpy(test_host,test_dev, 5 * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 5; i++)
    {
        std::cout << test_host[i] << std::endl;
    }
}

void nms_rotated_cuda(const float *dets_host_ptr, float *dets_dev_ptr, std::vector<int>& ids, const int max_nms_input_nums, const int box_nums, const float iou_threshold, const int top_k)
{
    if(!dets_dev_ptr)
    {
        HANDLE_ERROR(cudaMalloc((void **)&dets_dev_ptr, max_nms_input_nums * 5 * sizeof(float)));
    }
    HANDLE_ERROR(cudaMemcpy(dets_dev_ptr, dets_host_ptr, box_nums*5*sizeof(float), cudaMemcpyHostToDevice));


    const int col_blocks = DIVUP(box_nums,threadsPerBlock);
    dim3 blocks(col_blocks,col_blocks);
    dim3 threads(threadsPerBlock);


    unsigned long long *mask_host_ptr = new unsigned long long[box_nums * col_blocks];    
    unsigned long long *mask_dev_ptr = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&mask_dev_ptr,sizeof(unsigned long long) * box_nums * col_blocks));
    cudaMemset(mask_dev_ptr, 0, sizeof(unsigned long long) * box_nums * col_blocks);
    
    nms_rotated_cuda_kernel<float><<<blocks,threadsPerBlock>>>(dets_dev_ptr,box_nums,iou_threshold,mask_dev_ptr);
    
    //cudaDeviceSynchronize();
    // cudaMemcpy is a tongbu function , so no need cudaDeviceSynchronize
    HANDLE_ERROR(cudaMemcpy(mask_host_ptr,mask_dev_ptr, sizeof(unsigned long long)* box_nums * col_blocks, cudaMemcpyDeviceToHost)); //* box_nums * col_blocks

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0ULL, sizeof(unsigned long long) * col_blocks);
    int num_to_keep = 0;

    // for(int i = 0; i < box_nums; i++)
    // {
    //     int cur = i * col_blocks; 
    //     for(int j = 0; j < col_blocks; j++)
    //     {
    //         std::cout << mask_host_ptr[cur+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    bool break_flag = 0;
    for(int i = 0; i < box_nums; ++i)
    {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        // if(i == 2)
        // {
        //     std::bitset<64> bs(remv[nblock]);
        //     std::cout << bs << std::endl;
        // }

        if(!(remv[nblock] & (1ULL << inblock)))
        {
            ++num_to_keep;
            ids.push_back(i);
            unsigned long long* p = mask_host_ptr + i * col_blocks;
            for (int j = nblock; j < col_blocks; ++j) 
            {
                remv[j] |= p[j];
                if(num_to_keep == top_k)
                {
                    break_flag = 1;
                    break;
                }
            }
        }

        if(break_flag)
        {
            break;
        }
    }

    // for(int i = 0; i < ids.size(); i++)
    // {
    //     for(int j = 0; j < 5; j++)
    //     {
    //         std::cout << dets_host_ptr[ids[i] * 5 + j] << " " ; // << continuous_output[nms_ids[i] * 5 + j] << " ";
    //     }
    //     std::cout  << ids[i] << std::endl;
    // }

    delete mask_host_ptr;
    HANDLE_ERROR(cudaFree(mask_dev_ptr));
}
// 4 9
// 4 10