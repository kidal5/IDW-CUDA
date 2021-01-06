#pragma once

#include "CpuIdwBase.h"

/// Compute IDW on CPU using many threads
class CpuIdwThreaded final : public CpuIdwBase{
public:

	/*
	* Constructor
	* \param[in] _numOfThreads specifies how many threads should be used, -1 means all available
	*/
	CpuIdwThreaded(const int _width, const int _height, const int _numOfThreads = -1);


private:
	virtual void refreshInnerGreyscale(DataManager& manager) override;
	virtual void refreshInnerColor(const Palette& p) override;
	
	int numOfThreads = 1;
};
