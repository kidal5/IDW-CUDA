#pragma once

#include "CpuIdwBase.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class GpuIdwBase : public CpuIdwBase
{
public:
	GpuIdwBase(int _width, int _height, const std::string& _methodName);

	virtual ~GpuIdwBase() override;

	uint8_t* getBitmapGreyscaleCpu() override;
	uint32_t* getBitmapColorCpu() override;

	

protected:
	void refreshInnerGreyscale(DataManager& manager) override;
	void refreshInnerColor(const Palette& p) override;
	
	void copyAnchorsToGpu(const std::vector<P2>& anchorPoints);

	virtual void refreshInnerGreyscaleGpu(double pParam) = 0;
	virtual void refreshInnerColorGpu(const Palette& p) = 0;

	virtual void downloadGreyscaleBitmap() = 0;
	virtual void downloadColorBitmap() = 0;

	


protected:
	int anchorsGpuCurrentCount = -1;
	int anchorsGpuMaxCount = 100;
	size_t anchorsGpuBytes = anchorsGpuMaxCount * 3 * sizeof(int);
	int* anchorsGpu = nullptr;

	bool lastGreyscaleVersionOnCpu = false;
	bool lastColorVersionOnCpu = false;

};

