#pragma once

#include "CpuIdwBase.h"


/// Compute IDW using cpu / singlethreaded
class CpuIdw final : public CpuIdwBase{
public:
	
	/// Constructor
	CpuIdw(const int _width, const int _height) : CpuIdwBase(_width, _height, "CpuIdw") {}
	virtual ~CpuIdw() override = default;
private:
	virtual void refreshInnerGreyscale(DataManager& manager) override;

	virtual void refreshInnerColor(const Palette& p) override;
};
