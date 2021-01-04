#include "CpuIdwBase.h"


#include <chrono>
#include <utility>
#include <cmath>

#include "Utils.h"


CpuIdwBase::CpuIdwBase(const int _width, const int _height, std::string _methodName)
: width(_width), height(_height), methodName(std::move(_methodName)) {
	bitmapGreyscaleCpu = new uint8_t[width * height];
}

CpuIdwBase::~CpuIdwBase() {
	delete[] bitmapGreyscaleCpu;
}

int CpuIdwBase::getWidth() const {
	return static_cast<int>(width);
}

int CpuIdwBase::getHeight() const {
	return static_cast<int>(height);
}

std::string CpuIdwBase::getMethodName() const {
	return methodName;
}

float CpuIdwBase::getFps() const {
	return 1000000.0f / elapsedMicroseconds;
}

long long CpuIdwBase::getTimeInMilliseconds() const {
	return elapsedMicroseconds / 1000;
}

void CpuIdwBase::refresh(DataManager& manager, const bool forceRefresh) {
	if (!manager.getChange() && !forceRefresh) return;

	const auto timeBegin = std::chrono::system_clock::now();

	refreshInner(manager.getAnchorPoints(), manager.getPParam());
	refreshInnerDrawAnchorPoints(manager.getAnchorPoints());
	Utils::drawGui(manager, *this);
	elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - timeBegin).count();
}

void CpuIdwBase::refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) {

	//todo should only be in color space....
	
	for (const auto & point : anchorPoints) {
		//since on the specified point value is always 0, i take one pixel to the right...
		uint8_t value = bitmapGreyscaleCpu[point.y * width + point.x + 1] > 127 ? 0 : 255;

		for (int shiftX = -1; shiftX < 1; shiftX++) {
			for (int shiftY = -1; shiftY < 1; shiftY++) {
				bitmapGreyscaleCpu[(point.y + shiftY) * width + point.x + shiftX] = value;
				//bitmapCpu[4 * ((point.y + shiftY) * width + point.x + shiftX) + 0] = 255;
			}
		}
	}
}

uint8_t* CpuIdwBase::getBitmapGreyscaleCpu() {
	return bitmapGreyscaleCpu;
}

double CpuIdwBase::computeWiCpu(const P2& a, const P2& b, const double p) {
	const auto dist = (a - b).norm2d();
	return 1 / pow(dist, p);
}