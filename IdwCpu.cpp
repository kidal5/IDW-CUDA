#include "IdwCpu.h"

#include <iostream>
#define LOG(x) std::cout << x << std::endl

IdwCpu::IdwCpu(const int _width, const int _height) : width(_width), height(_height) {

	bitmap = std::unique_ptr<uint8_t[]>(new uint8_t[width * height]);
	std::fill_n(bitmap.get(), width * height, 0);
	
}


int count = 0;

void IdwCpu::refresh() {

	count = count + 25;
	bitmap[count] = 255;


	//zero zero for opengl is left/bottom
	//it goes line by line
	//
	LOG('a');
	for (int h = 0; h < height; ++h) {
		bitmap[h * height + 10] = 255;
	}


	
}
