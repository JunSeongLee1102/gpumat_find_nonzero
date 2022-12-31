/**
 * @file    Types.cuh
 *
 * @author  btran
 *
 */

#define OVERLAP 0
#pragma once

#include <thrust/device_vector.h>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "cuda_utils.cuh"

enum FIND_NONZERO_MODE {
    CPU,
    CPU_DOWNLOAD,
    GPU,
    GPU_VALIDATION
};


struct PointXY
{
    int x = 0;
    int y = 0;
};

struct PointIndex {
    int noise;
    int cluster;
};











