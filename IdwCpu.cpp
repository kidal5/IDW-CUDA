#include "IdwCpu.h"

#include <iostream>

#define LOG(x) std::cout << x << std::endl

IdwCpu::IdwCpu(const int _width, const int _height) : width(_width), height(_height) {
	bitmap = std::unique_ptr<uint8_t[]>(new uint8_t[3 * width * height]);
	std::fill_n(bitmap.get(), 3 * width * height, 0);
}


void IdwCpu::refresh(std::vector<P3> & anchorPoints) const {

	auto computeWi = [](const P3& a, const P3& b, const double p = 10) {
		const auto dist = (a - b).norm2d();
		return 1 / pow(dist, p);
	};
	

	std::fill_n(bitmap.get(), 3 * width * height, 0);


	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			double wiSum = 0;
			double outputSum = 0;
			
			for (const auto & point: anchorPoints) {

				const double wi = computeWi({w,h, 0}, point);
				wiSum += wi;
				outputSum += wi * point.z;
			}
			outputSum /= wiSum;
			bitmap[3 * (h * height + w) + 2] = static_cast<uint8_t>(outputSum);
		}
		

	}
}

