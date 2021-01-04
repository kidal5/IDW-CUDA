#pragma once

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
	virtual uint32_t* getBitmapColorCpu();

	void drawOpengl(DataManager& manager);
	
	static double computeWiCpu(const P2& a, const P2& b, const double p = 10);

private:
	virtual void refreshInnerGreyscale(DataManager& manager) = 0;
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints);

	virtual void refreshInnerColor(const Palette & p) = 0;
	
protected:
	long long width;
	long long height;
	std::string methodName;
	uint8_t* bitmapGreyscaleCpu;
	uint32_t* bitmapColorCpu;

	long long elapsedMicroseconds = 1;

};

