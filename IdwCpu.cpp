#include "IdwCpu.h"

void IdwCpu::refreshInner(const std::vector<P2>& anchorPoints, const double pParam){

	clearBitmap();
	
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			double wiSum = 0;
			double outputSum = 0;

			for (const auto& point : anchorPoints) {

				const double wi = computeWiCpu({ w,h, 0 }, point, pParam);
				wiSum += wi;
				outputSum += wi * point.value;
			}
			outputSum /= wiSum;
			bitmapCpu[3 * (h * width + w) + 2] = static_cast<uint8_t>(outputSum);
		}
	}
}