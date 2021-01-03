#pragma once

#include "CpuIdwBase.h"


class CpuIdw final : public CpuIdwBase{
public:
	CpuIdw(const int _width, const int _height) : CpuIdwBase(_width, _height, "CpuIdw") {}
	virtual ~CpuIdw() override = default;
private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints, double pParam) override;
};
