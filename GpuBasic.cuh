#pragma once

#include "GpuIdwBase.cuh"

class GpuBasic final : public GpuIdwBase{
public:
	GpuBasic(const int _width, const int _height) : GpuIdwBase(_width, _height, "GpuBasic") {}


protected:
	virtual void refreshInnerGpu(double pParam) override;
};
