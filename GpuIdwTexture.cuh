#pragma once

#include "GpuIdwBase.cuh"
#include "cuda_runtime.h"
//#include "cuda_texture_types.h"
//#include "device_launch_parameters.h"

class GpuIdwTexture final : public GpuIdwBase {

public:
	GpuIdwTexture(int _width, int _height);

	virtual ~GpuIdwTexture() override;



protected:
	virtual void refreshInnerGreyscaleGpu(double pParam) override;
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;


	virtual void refreshInnerColorGpu(const Palette& p) override;
	virtual void downloadGreyscaleBitmap() override;
	virtual void downloadColorBitmap() override;
private:
	cudaArray* cuArrayGreyscale = nullptr;
	cudaArray* cuArrayColor = nullptr;
	cudaSurfaceObject_t surfObject = 0;

};
