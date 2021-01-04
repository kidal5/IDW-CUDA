#include "CpuIdw.h"

void CpuIdw::refreshInnerGreyscale(DataManager& manager){

	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			double wiSum = 0;
			double outputSum = 0;

			
			for (const auto& point : manager.getAnchorPoints()) {

				const double wi = computeWiCpu({ w,h, 0 }, point, manager.getPParam());
				wiSum += wi;
				outputSum += wi * point.value;
			}
			outputSum /= wiSum;

			bitmapGreyscaleCpu[h * width + w] = static_cast<uint8_t>(outputSum);
		}
	}
}

void CpuIdw::refreshInnerColor(const Palette& p) {
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			const uint32_t val = bitmapGreyscaleCpu[h * width + w];
			const auto output = static_cast<uint32_t>(val << 24 || val << 16 || val << 8 || val);

			bitmapColorCpu[h * width + w] = 0xFF000000;
		}
	}
	
}
