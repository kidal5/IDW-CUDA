#pragma once

#include <memory>

#include <vector>

class IdwCpu {
public:

	IdwCpu(int _width, int _height);
	
	void refresh();

	int width;
	int height;

	std::unique_ptr<uint8_t[]> bitmap;

};
