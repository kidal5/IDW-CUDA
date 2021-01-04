#pragma once

#include <memory>
#include <vector>

#include "DataManager.h"
#include "P2.h"

#include <string>
#include <fmt/core.h>

class CpuIdwBase {
public:
	CpuIdwBase(int _width, int _height, std::string _methodName);
	virtual ~CpuIdwBase();

	
	int getWidth() const;
	int getHeight() const;
	std::string getMethodName() const;

	float getFps() const;
	long long getTimeInMilliseconds() const;

	void refresh(DataManager& manager, bool forceRefresh = false);
	virtual uint8_t* getBitmapGreyscaleCpu();

	static double computeWiCpu(const P2& a, const P2& b, const double p = 10);

private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints, double pParam) = 0;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints);
	
protected:
	long long width;
	long long height;
	std::string methodName;
	uint8_t* bitmapGreyscaleCpu;

	long long elapsedMicroseconds = 1;

};

