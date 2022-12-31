#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_utils.cuh"
#include "types.cuh"


//계속 같은 크기 array 들어온다 가정하고 메모리 재활용
//GpuMat의 nonzero element 추출하는 class
class GpuMatFindNonzero
{
public:
	GpuMatFindNonzero(int rows, int cols, int step, int max_num_nonzeros, cudaStream_t* stream=0);
	~GpuMatFindNonzero();

	void findNonzero(cv::cuda::GpuMat, bool is_validation = false);

	int      num_nonzeros_;
	PointXY* gpu_nonzero_xy_coords_;
	//검증용
	PointXY* cpu_nonzero_xy_coords_;
private:
	cudaStream_t* stream_;
	int rows_;
	int cols_;
	int step_;
	int max_num_nozneros_;

	int row_block_size_;
	int col_block_size_;

	dim3 block_;
	dim3 grid_;
	int* cpu_num_nonzeros_;
	int* gpu_num_nonzeros_;

};







