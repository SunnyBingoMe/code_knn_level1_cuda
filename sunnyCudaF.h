#ifndef _SUNNY_CUDA_F_
#define _SUNNY_CUDA_F_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

void cudaMemsetFloat_sunny(float *pd_data, float value, int count){
    thrust::device_ptr<float> dev_ptr(pd_data);
    thrust::fill(dev_ptr, dev_ptr + count, value);
}

/* general define for cuda,  OBS: maybe not suitable for 3D */
#define Grid_width      gridDim.x
#define Grid_height     gridDim.y
#define Grid_pages      gridDim.z
#define OneGrid_size1D  Grid_width
#define OneGrid_size2D  (Grid_width * Grid_height)
#define OneGrid_size3D  (Grid_width * Grid_height * Grid_pages)
#define OneGrid_size    OneGrid_size3D
#define Block_width         blockDim.x
#define Block_height        blockDim.y
#define Block_pages         blockDim.z
#define OneBlock_size2D     (Block_width * Block_height)
#define OneBlock_size3D     (Block_width * Block_height * Block_pages)
#define OneBlock_size       OneBlock_size3D
#endif
