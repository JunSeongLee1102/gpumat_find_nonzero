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


//��� ���� ũ�� array ���´� �����ϰ� �޸� ��Ȱ��
//GpuMat�� nonzero element �����ϴ� class
class GpuMatFindNonzero
{
public:
	GpuMatFindNonzero(int rows, int cols, int step, int max_num_nonzeros, cudaStream_t* stream=0);
	~GpuMatFindNonzero();

	void findNonzero(cv::cuda::GpuMat, bool is_validation = false);
	//NonzeroCoords* _nonzero_coords;
	int      _num_nonzeros;
	PointXY* _gpu_nonzero_xy_coords;
	//������
	PointXY* _cpu_nonzero_xy_coords;
private:
	cudaStream_t* _stream;
	int _rows;
	int _cols;
	int _step;
	int _max_num_nonzeros;

	int _row_block_size;
	int _col_block_size;

	dim3 _block;
	dim3 _grid;
	int* _cpu_num_nonzeros;
	int* _gpu_num_nonzeros;

};







