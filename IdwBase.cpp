#include "IdwBase.h"


#include <chrono>
#include <utility>
#include <cmath>


IdwBase::IdwBase(const int _width, const int _height, std::string _methodName)
: width(_width), height(_height), methodName(std::move(_methodName)) {
	bitmapCpu = std::unique_ptr<uint8_t[]>(new uint8_t[3 * width * height]);
}

int IdwBase::getWidth() const {
	return static_cast<int>(width);
}

int IdwBase::getHeight() const {
	return static_cast<int>(height);
}

void IdwBase::refresh(AnchorPointsManager& manager) {
	if (!manager.getChange()) return;

	long long elapsed;
	refresh(manager, elapsed);
	fmt::print("Time for {:<15} is {:8} milliseconds / {:5f} FPS\n", methodName, elapsed, 1.0 / elapsed);
}


void IdwBase::refresh(AnchorPointsManager& manager, long long& elapsedMilliseconds) {
	if (!manager.getChange()) return;

	const auto timeBegin = std::chrono::system_clock::now();

	refreshInner(manager.getAnchorPoints());
	refreshInnerDrawAnchorPoints(manager.getAnchorPoints());

	elapsedMilliseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - timeBegin).count();
}

void IdwBase::refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) {
	for (const auto& point : anchorPoints) {
		//set to max
		bitmapCpu[3 * (point.y * width + point.x) + 0] = 255;
		bitmapCpu[3 * (point.y * width + point.x) + 1] = 255;
		bitmapCpu[3 * (point.y * width + point.x) + 2] = 255;
	}
}

void* IdwBase::getBitmapCpu() {
	return bitmapCpu.get();
}

double IdwBase::computeWiCpu(const P2& a, const P2& b, const double p) {
	const auto dist = (a - b).norm2d();
	return 1 / pow(dist, p);
}

void IdwBase::clearBitmap() {
	std::memset(bitmapCpu.get(), 0, sizeof(uint8_t) * 3 * width * height);
	//std::fill_n(bitmapCpu.get(), 3 * width * height, 0);
}
