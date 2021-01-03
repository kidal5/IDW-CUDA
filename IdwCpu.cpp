#include "IdwCpu.h"


IdwCpu::IdwCpu(const int _width, const int _height) : IdwBase(_width, _height,  "IdwCpu") {
	bitmap = std::unique_ptr<uint8_t[]>(new uint8_t[3 * width * height]);
}


void IdwCpu::refreshInner(const std::vector<P2>& anchorPoints) const {

	auto computeWi = [](const P2& a, const P2& b, const double p = 10) {
		const auto dist = (a - b).norm2d();
		return 1 / pow(dist, p);
	};

	std::fill_n(bitmap.get(), 3 * width * height, 0);

	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			double wiSum = 0;
			double outputSum = 0;

			for (const auto& point : anchorPoints) {

				const double wi = computeWi({ w,h, 0 }, point);
				wiSum += wi;
				outputSum += wi * point.value;
			}
			outputSum /= wiSum;
			bitmap[3 * (h * height + w) + 2] = static_cast<uint8_t>(outputSum);
		}
	}

}

void IdwCpu::refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) const {
	for (const auto& point : anchorPoints) {
		//set to max
		bitmap[3 * (point.y * height + point.x) + 0] = 255;
		bitmap[3 * (point.y * height + point.x) + 1] = 255;
		bitmap[3 * (point.y * height + point.x) + 2] = 255;
	}
}

void* IdwCpu::getBitmapCpu() {
	return bitmap.get();
}