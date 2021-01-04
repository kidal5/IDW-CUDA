#pragma once

#include "GpuIdwBase.cuh"
#include "cuda_runtime.h"
//#include "cuda_texture_types.h"
//#include "device_launch_parameters.h"

class GpuIdwTexture final : public GpuIdwBase {

public:
	GpuIdwTexture(int _width, int _height);

	virtual ~GpuIdwTexture() override;

	uint8_t* getBitmapCpu() override;


protected:
	virtual void refreshInnerGpu(double pParam) override;
private:
	cudaArray* cuArray = nullptr;
	cudaSurfaceObject_t surfObject = 0;

};
