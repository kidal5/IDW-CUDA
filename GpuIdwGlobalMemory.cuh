#pragma once

#include "GpuIdwBase.cuh"

class GpuIdwGlobalMemory final : public GpuIdwBase {
public:
	GpuIdwGlobalMemory(const int _width, const int _height) : GpuIdwBase(_width, _height, "GpuIdwGlobalMemory") {}


protected:
	virtual void refreshInnerGpu(double pParam) override;
};
