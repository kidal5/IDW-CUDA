#pragma once

#include "CpuIdwBase.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
* Little helper class for using IDW with CUDA
* 
* manages everything about anchor data
* aka sends them into graphics
*/
class GpuIdwBase : public CpuIdwBase
{
public:
	/// Constructor
	GpuIdwBase(int _width, int _height, const std::string& _methodName);

	virtual ~GpuIdwBase() override;
	
	/// Overrided because data must be downloaded to cpu
	uint8_t* getBitmapGreyscaleCpu() override;

	/// Overrided because data must be downloaded to cpu
	uint32_t* getBitmapColorCpu() override;

protected:

	void refreshInnerGreyscale(DataManager& manager) override;
	void refreshInnerColor(const Palette& p) override;
	
	void copyAnchorsToGpu(const std::vector<P2>& anchorPoints);
	void copyPaletteToGpu(const Palette& p);

	/// override me / refresh greyscale data
	virtual void refreshInnerGreyscaleGpu(double pParam) = 0;

	/// override me / refresh color data
	virtual void refreshInnerColorGpu() = 0;

	/// override me / download greyscale data from gpu to cpu
	virtual void downloadGreyscaleBitmap() = 0;
	
	/// override me / download color data from gpu to cpu
	virtual void downloadColorBitmap() = 0;


protected:
	int anchorsGpuCurrentCount = -1;
	int anchorsGpuMaxCount = 1501;
	size_t anchorsGpuBytes = anchorsGpuMaxCount * 3 * sizeof(int);
	int* anchorsGpu = nullptr;

	bool lastGreyscaleVersionOnCpu = false;
	bool lastColorVersionOnCpu = false;

	uchar4* colorMappingData;
	std::string lastPaletteName = "";

	dim3 gridRes;
	dim3 blockRes;

	dim3 gridAnchorRes;
	dim3 blockAnchorRes = dim3(256);
	
};

