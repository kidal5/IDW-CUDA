#pragma once

#include "CpuIdwBase.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class GpuIdwBase : public CpuIdwBase
{
public:
	GpuIdwBase(int _width, int _height, const std::string& _methodName);

	virtual ~GpuIdwBase() override;


protected:
	void refreshInner(const std::vector<P2>& anchorPoints, double pParam) override;
	void copyAnchorsToGpu(const std::vector<P2>& anchorPoints);

	virtual void refreshInnerGpu(double pParam) = 0;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;



protected:
	uint8_t* bitmapGpu = nullptr;
	int anchorsGpuCurrentCount = -1;
	int anchorsGpuMaxCount = 100;
	size_t anchorsGpuBytes = anchorsGpuMaxCount * 3 * sizeof(int);
	int* anchorsGpu = nullptr;

	bool lastVersionOnCpu = false;


};

