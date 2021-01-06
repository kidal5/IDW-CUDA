#pragma once

#include <vector>

#include "DataManager.h"
#include "P2.h"

#include <string>
#include <fmt/core.h>
#include <fmt/color.h>

/// OpenGL constatnt redefinition
const int GL_UNSIGNED_INT_8_8_8_8 = 0x8035;

/// OpenGL constatnt redefinition
const int GL_UNSIGNED_INT_8_8_8_8_REV = 33639;

/*
* Base class for any IDW computor method
* All classes must override atleast refreshInnerGreyscale and refreshInnerColor methods 
* 
* This class manages bitmap data
*/
class CpuIdwBase {
public:
	
	//Constructor
	CpuIdwBase(int _width, int _height, std::string _methodName, bool _isCpuKernel = true);
	virtual ~CpuIdwBase();

	/// get width of image
	int getWidth() const;
	
	/// get height of image
	int getHeight() const;
	
	/// get name of method
	std::string getMethodName() const;
	
	/// returns true when method is CPU based and thus very slow
	bool isCpuKernel() const;

	/// combined width and height into one variable
	P2 getImgSize() const;

	/// get fps for current method
	float getFps() const;
	
	/// get time for one image generation
	long long getTimeInMilliseconds() const;

	/// generate new image
	void refresh(DataManager& manager, bool forceRefresh = false);
	
	/// retrun cpu greyscale data
	virtual uint8_t* getBitmapGreyscaleCpu();

	/// retrun cpu color data
	virtual uint32_t* getBitmapColorCpu();

	/// send current image into OpenGl 
	virtual void drawOpengl(DataManager& manager);
	
	/// compute Wi for used in IDW computation
	static double computeWiCpu(const P2& a, const P2& b, const double p = 10);

	/// compute 4 uint8_t into one uint32_t
	static uint32_t pack(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
		return (c0 << 24) | (c1 << 16) | (c2 << 8) | c3;
	}


private:
	/// override me / refresh greyscale image
	virtual void refreshInnerGreyscale(DataManager& manager) = 0;

	/// override me / draw anchor points into greyscale image
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints);

	/// override me / create color image from greyscale image
	virtual void refreshInnerColor(const Palette & p) = 0;
	
protected:
	long long width;
	long long height;
	std::string methodName;
	uint8_t* bitmapGreyscaleCpu;
	uint32_t* bitmapColorCpu;

	long long elapsedMicroseconds = 1;

	bool isCpuKernelInner;
	
};

