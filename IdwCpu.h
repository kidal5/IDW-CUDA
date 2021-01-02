#pragma once

#include <memory>
#include <vector>
#include "P2.h"


class IdwCpu {
public:

	IdwCpu(int _width, int _height);
	
	void refresh(std::vector<P2>& anchorPoints, bool change) const;

	int width;
	int height;

	std::unique_ptr<uint8_t[]> bitmap;

};
