#pragma once

#include "CpuIdwBase.h"

class CpuIdwThreaded final : public CpuIdwBase{
public:

	CpuIdwThreaded(const int _width, const int _height, const int _numOfThreads = -1);


private:
	virtual void refreshInnerGreyscale(DataManager& manager) override;
	virtual void refreshInnerColor(const Palette& p) override;
	
	int numOfThreads = 1;
};
