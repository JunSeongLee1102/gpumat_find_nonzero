#include "gpumat_find_nonzero.cuh"
#include <cmath>



__global__  
void kernelFindNonzero(unsigned char* gpu_image_data, int rows, int cols, int step,
	PointXY* nonzero_xy_coords, int* gpu_num_nonzeros)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y; //Row number
	int j = blockIdx.x * blockDim.x + threadIdx.x; //Column number

	int cur_ind = 0;
	//이미지 범위 내의 인덱스 --> 0이 아닐 경우 nonzero에 저장
	//atomicAdd를 통한 중복 저장이나 메모리 동시 엑세스 방지
	if (i < rows && j < cols)
	{
		if (gpu_image_data[i * step + j] != unsigned char(0))
		{
			cur_ind = atomicAdd(&gpu_num_nonzeros[0], 1);
			nonzero_xy_coords[cur_ind].x = j;
			nonzero_xy_coords[cur_ind].y = i;
		}

	}
}

GpuMatFindNonzero::GpuMatFindNonzero(int rows, int cols, int step, int max_num_nonzeros, cudaStream_t* stream)
{
	rows_ = rows;
	cols_ = cols;
	step_ = step;
	max_num_nozneros_ = max_num_nonzeros;

	//consider warp thread size
	row_block_size_ = int(rows/32+1);
	col_block_size_ = int(cols/32+1);

	grid_ = dim3(col_block_size_, row_block_size_);
	block_  = dim3(32, 32);

	cpu_num_nonzeros_ = new int[1];
	cpu_num_nonzeros_[0] = 0;
	cpu_nonzero_xy_coords_ = nullptr;

	stream_ = stream;

	HANDLE_ERROR(cudaMallocAsync((void**)&gpu_num_nonzeros_, sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMemsetAsync(gpu_num_nonzeros_, 0, sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&gpu_nonzero_xy_coords_, 2 * max_num_nozneros_ * sizeof(int), stream_[0]));

	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));
}

GpuMatFindNonzero::~GpuMatFindNonzero()
{
	HANDLE_ERROR(cudaFreeAsync(gpu_num_nonzeros_, stream_[0]));
	HANDLE_ERROR(cudaFreeAsync(gpu_nonzero_xy_coords_, stream_[0]));

	delete[] cpu_num_nonzeros_;

	if (cpu_nonzero_xy_coords_ != nullptr)
		delete[] cpu_nonzero_xy_coords_;

	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));
}

void GpuMatFindNonzero::findNonzero(cv::cuda::GpuMat gpu_image, bool is_validation)
{	
	HANDLE_ERROR(cudaMemsetAsync(gpu_num_nonzeros_, 0, sizeof(int), stream_[0]));

	kernelFindNonzero <<<grid_, block_, 0, stream_[0]>>> (reinterpret_cast<unsigned char*>(gpu_image.data), rows_, 
														 cols_, step_, gpu_nonzero_xy_coords_, gpu_num_nonzeros_);

	HANDLE_ERROR(cudaMemcpyAsync(&cpu_num_nonzeros_[0], &gpu_num_nonzeros_[0], sizeof(int), 
						      cudaMemcpyDeviceToHost, stream_[0]));
	
	if (is_validation)
	{
		HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));
		if (cpu_nonzero_xy_coords_ != nullptr)
			delete[] cpu_nonzero_xy_coords_;
		cpu_nonzero_xy_coords_ = new PointXY[cpu_num_nonzeros_[0]];
		HANDLE_ERROR(cudaMemcpyAsync(&cpu_nonzero_xy_coords_[0], &gpu_nonzero_xy_coords_[0], cpu_num_nonzeros_[0] * sizeof(PointXY), cudaMemcpyDeviceToHost, stream_[0]));
	}

	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));

	num_nonzeros_ = cpu_num_nonzeros_[0];
	return;
}