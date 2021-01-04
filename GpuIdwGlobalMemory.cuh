#pragma once

#include "GpuIdwBase.cuh"

class GpuIdwGlobalMemory final : public GpuIdwBase {
public:
	GpuIdwGlobalMemory(const int _width, const int _height);

	virtual ~GpuIdwGlobalMemory() override;

	uint8_t* getBitmapCpu() override;

protected:
	virtual void refreshInnerGpu(double pParam) override;


private:
	size_t imgBytesCount = -1;
};
