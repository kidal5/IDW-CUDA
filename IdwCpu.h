#pragma once

#include <memory>
#include <vector>
#include "P3.h"


class IdwCpu {
public:

	IdwCpu(int _width, int _height);
	
	void refresh(std::vector<P3>& anchorPoints) const;

	int width;
	int height;

	std::unique_ptr<uint8_t[]> bitmap;

};
