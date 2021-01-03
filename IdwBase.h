#pragma once

#include <memory>
#include <vector>

#include "AnchorPointsManager.h"
#include "P2.h"

#include <string>
#include <fmt/core.h>

class IdwBase {
public:
	virtual ~IdwBase() = default;
	IdwBase(int _width, int _height, std::string _methodName);

	int getWidth() const;
	int getHeight() const;
	std::string getMethodName() const;

	float getFps() const;
	long long getTimeInMilliseconds() const;

	void refresh(AnchorPointsManager& manager);
	virtual uint8_t* getBitmapCpu();

	static double computeWiCpu(const P2& a, const P2& b, const double p = 10);

protected:
	void clearBitmap();

	
private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints, double pParam) = 0;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints);
	
protected:
	long long width;
	long long height;
	std::string methodName;
	std::unique_ptr<uint8_t[]> bitmapCpu;

	long long elapsedMicroseconds = 1;

};

