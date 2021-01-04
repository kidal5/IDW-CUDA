#pragma once

#include "CpuIdwBase.h"


class CpuIdw final : public CpuIdwBase{
public:
	CpuIdw(const int _width, const int _height) : CpuIdwBase(_width, _height, "CpuIdw") {}
	virtual ~CpuIdw() override = default;
private:
	virtual void refreshInnerGreyscale(DataManager& manager) override;

	virtual void refreshInnerColor(const Palette& p) override;
};
