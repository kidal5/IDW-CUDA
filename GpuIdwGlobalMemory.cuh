#pragma once

#include "GpuIdwBase.cuh"

class GpuIdwGlobalMemory final : public GpuIdwBase {
public:
	GpuIdwGlobalMemory(const int _width, const int _height);

	virtual ~GpuIdwGlobalMemory() override;

protected:
	virtual void refreshInnerGreyscaleGpu(double pParam) override;
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;


	virtual void refreshInnerColorGpu(const Palette& p) override;
	virtual void downloadGreyscaleBitmap() override;
	virtual void downloadColorBitmap() override;
	
private:
	size_t bitmapGreyscaleBytesCount = -1;
	size_t bitmapColorBytesCount = -1;

	uint8_t* bitmapGreyscaleGpu = nullptr;
	uint32_t* bitmapColorGpu = nullptr;

};
