#include "CpuIdwBase.h"


#include <chrono>
#include <utility>
#include <cmath>


CpuIdwBase::CpuIdwBase(const int _width, const int _height, std::string _methodName)
: width(_width), height(_height), methodName(std::move(_methodName)) {
	bitmapCpu = std::unique_ptr<uint8_t[]>(new uint8_t[4 * width * height]);
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

void CpuIdwBase::refresh(AnchorPointsManager& manager, const bool forceRefresh) {
	if (!manager.getChange() && !forceRefresh) return;

	const auto timeBegin = std::chrono::system_clock::now();

	refreshInner(manager.getAnchorPoints(), manager.getPParam());
	refreshInnerDrawAnchorPoints(manager.getAnchorPoints());

	elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - timeBegin).count();
}

void CpuIdwBase::refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) {

	for (const auto & point : anchorPoints) {
		for (int shiftX = -1; shiftX < 1; shiftX++) {
			for (int shiftY = -1; shiftY < 1; shiftY++) {
				bitmapCpu[4 * ((point.y + shiftY) * width + point.x + shiftX) + 0] = 255;
				bitmapCpu[4 * ((point.y + shiftY) * width + point.x + shiftX) + 1] = 0;
				bitmapCpu[4 * ((point.y + shiftY) * width + point.x + shiftX) + 2] = 0;
				bitmapCpu[4 * ((point.y + shiftY) * width + point.x + shiftX) + 3] = 0;
			}
		}
	}
}

uint8_t* CpuIdwBase::getBitmapCpu() {
	return bitmapCpu.get();
}

double CpuIdwBase::computeWiCpu(const P2& a, const P2& b, const double p) {
	const auto dist = (a - b).norm2d();
	return 1 / pow(dist, p);
}

void CpuIdwBase::clearBitmap() {
	std::memset(bitmapCpu.get(), 0, sizeof(uint8_t) * 4 * width * height);
	//std::fill_n(bitmapCpu.get(), 3 * width * height, 0);
}
