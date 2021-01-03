#pragma once

#include "CpuIdwBase.h"

class CpuIdwThreaded final : public CpuIdwBase{
public:

	CpuIdwThreaded(const int _width, const int _height, const int _numOfThreads = -1);


private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints, double pParam) override;

private:
	int numOfThreads = 1;
};
