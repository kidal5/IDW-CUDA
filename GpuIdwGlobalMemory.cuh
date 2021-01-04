#pragma once

#include "GpuIdwBase.cuh"

class GpuIdwGlobalMemory final : public GpuIdwBase {
public:
	GpuIdwGlobalMemory(const int _width, const int _height);

	virtual ~GpuIdwGlobalMemory() override;

	uint8_t* getBitmapGreyscaleCpu() override;

protected:
	virtual void refreshInnerGpu(double pParam) override;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;


private:
	size_t imgBytesCount = -1;
};
