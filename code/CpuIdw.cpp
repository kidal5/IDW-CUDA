#include "CpuIdw.h"

void CpuIdw::refreshInnerGreyscale(DataManager& manager){

	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			wiSum = 0;
			outputSum = 0;

			for anchorPoint in anchorPoints {
			    dist = computeDistance(currentPoint, anchorPoint);

				wiSum += wi =  1 / pow(dist, p);
				outputSum += wi * point.value;
			}

			output[h,w]=outputSum / wiSum;
		}
	}
}


void CpuIdw::refreshInnerColor(const Palette& p) {
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			const uint8_t val = bitmapGreyscaleCpu[h * width + w];
			bitmapColorCpu[h * width + w] = p.colorData[val];
		}
	}
	
}
