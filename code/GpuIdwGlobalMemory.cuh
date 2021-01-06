#pragma once

#include "GpuIdwBase.cuh"

/**
* IDW computor method that uses CUDA global memory as memory backend
*/
class GpuIdwGlobalMemory final : public GpuIdwBase {
public:
	/// Constructor
	GpuIdwGlobalMemory(const int _width, const int _height);

	virtual ~GpuIdwGlobalMemory() override;

protected:
	virtual void refreshInnerGreyscaleGpu(double pParam) override;
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;


	virtual void refreshInnerColorGpu() override;
	virtual void downloadGreyscaleBitmap() override;
	virtual void downloadColorBitmap() override;
	
private:
	size_t bitmapGreyscaleBytesCount = -1;
	size_t bitmapColorBytesCount = -1;

	uint8_t* bitmapGreyscaleGpu = nullptr;
	uchar4* bitmapColorGpu = nullptr;

};
