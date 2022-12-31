#pragma once
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#define _USE_MATH_DEFINES

#include <fstream>
#include <thread>
#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "gpumat_find_nonzero.cuh"
#include "types.cuh"

using namespace std;
using std::cout;
using std::endl;

using std::string;

const int img_size = 5000;



using cv::Mat;
using cv::cuda::GpuMat;


//iteration 횟수 설정 및 FIND_NONZERO_MODE 설정: CPU, CPU_DOWNLOAD, GPU, GPU_VALIDATION
const int num_iterations = 11;
FIND_NONZERO_MODE mode_find_nonzero = FIND_NONZERO_MODE::GPU;


void display(Mat image, int x_size = 1000, int y_size = 1000)
{
    Mat vis_image;
    cv::resize(image, vis_image, cv::Size(x_size, y_size), 0, 0, cv::INTER_AREA);
    cv::imshow("Sample", vis_image);
    cv::waitKey();
}

void main()
{
    std::vector<double> times;
    std::chrono::system_clock::time_point start;

    cv::cuda::Stream* cv_stream=nullptr;
    cudaStream_t* stream=nullptr;
    cv::cuda::GpuMat gpu_sample_img;
    GpuMatFindNonzero* gpumat_find_nonzero=nullptr;

    cv::Mat sample_img(img_size, img_size, CV_8UC1, cv::Scalar(0));
    cv::Rect rect = cv::Rect(0, 0, img_size / 10, img_size / 10);
    cv::rectangle(sample_img, cv::Rect(0, 0, img_size / 10, img_size / 10), cv::Scalar(255), -1);
    cv::Mat sample_nonzeros;

    if (mode_find_nonzero != FIND_NONZERO_MODE::CPU)
    {
        cv_stream = new cv::cuda::Stream();
        stream = new cudaStream_t();
        stream[0] = cv::cuda::StreamAccessor::getStream(cv_stream[0]);
        gpu_sample_img.upload(sample_img, cv_stream[0]);
        cv_stream[0].waitForCompletion();
        if (mode_find_nonzero != FIND_NONZERO_MODE::CPU_DOWNLOAD)
        {
            gpumat_find_nonzero = new GpuMatFindNonzero(gpu_sample_img.rows, gpu_sample_img.cols, gpu_sample_img.step,
                (img_size / 10 + 1) * (img_size / 10 + 1), stream);
        }
    }
           
    for (int i = 0; i < num_iterations; i++)
    {
        //GPU initialization 시간 계산에서 제외(지속적으로 이미지 처리되고 있을 때의 시간 기준으로 비교)
        if(i==1)
            start = std::chrono::system_clock::now();  
        
        switch (mode_find_nonzero)
        {
            case(FIND_NONZERO_MODE::CPU_DOWNLOAD):
            {
                gpu_sample_img.download(sample_img, cv_stream[0]);
                cv_stream[0].waitForCompletion();
            }
            case(FIND_NONZERO_MODE::CPU):
            {
                cv::findNonZero(sample_img, sample_nonzeros);
                break;
            }
            case(FIND_NONZERO_MODE::GPU):
            {
                gpumat_find_nonzero->findNonzero(gpu_sample_img);
                break;
            }
            case(FIND_NONZERO_MODE::GPU_VALIDATION):
            {
                gpumat_find_nonzero->findNonzero(gpu_sample_img, true);
                cv::Mat gpu_result_store(img_size, img_size, CV_8UC1, cv::Scalar(0));
                
                //GpuMat find nonzero 결과 옮기기
                for (int j = 0; j < gpumat_find_nonzero->num_nonzeros_; j++)
                {
                    gpu_result_store.at<unsigned char>(gpumat_find_nonzero->cpu_nonzero_xy_coords_[j].y, gpumat_find_nonzero->cpu_nonzero_xy_coords_[j].x) = unsigned char(255);
                }

                //원본 이미지와 일치 여부 판정
                for (int jy = 0; jy < img_size; jy++)
                {
                    for (int kx = 0; kx < img_size; kx++)
                    {
                        if (sample_img.at<unsigned char>(jy, kx) != gpu_result_store.at<unsigned char>(jy, kx))
                            std::cout << "일치하지 않음!\n" << std::endl;
                    }
                }
                break;
            }
        }       
    }

    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "총 이미지 처리 시간: " << sec.count() << "seconds" << std::endl;

    double sec_per_image = sec.count() / (num_iterations-1);
    std::cout << "이미지 당 평균 처리 시간 : " << sec_per_image*1000.0 << " milli seconds" << std::endl;

    times.push_back(sec_per_image);



 
    return;    
}