#include "IdwCpu.h"

#include <iostream>
#define LOG(x) std::cout << x << std::endl

IdwCpu::IdwCpu(const int _width, const int _height) : width(_width), height(_height) {

	bitmap = std::unique_ptr<uint8_t[]>(new uint8_t[3 * width * height]);
	std::fill_n(bitmap.get(), 3 * width * height, 0);

}


int count = 1;

void IdwCpu::refresh() {

	//zero zero for opengl is left/bottom
	//it goes line by line

	std::fill_n(bitmap.get(), 3 * width * height, 0);
	count++;

	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {

			if (w % count == 0)
				bitmap[3 * (h * height + w)] = 255;

			if (w % count == 1)
				bitmap[3 * (h * height + w) + 1] = 255;

			if (w % count == 2)
				bitmap[3 * (h * height + w) + 2] = 255;

			
			
		}

	}


	
}
