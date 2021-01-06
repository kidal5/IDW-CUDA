#pragma once

#include "GpuIdwBase.cuh"
#include "cuda_runtime.h"

/**
* IDW computor method that uses CUDA texture as data backend
* 
* Can work in two modes based on data transfers between RAM and VRAM
* 
* first mode must copy data into RAM in order to show them on screen
* second mode does not. And thus is much faster with lower GPU utilization
*/
class GpuIdwTexture final : public GpuIdwBase {

public:
	/*
	* Constructor
	* _useOpenGlInterop defines which datatransfer method is used
	*/
	GpuIdwTexture(int _width, int _height, bool _useOpenGLInterop);

	virtual ~GpuIdwTexture() override;

protected:
	virtual void refreshInnerGreyscaleGpu(double pParam) override;
	virtual void refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) override;

	virtual void refreshInnerColorGpu() override;
	virtual void downloadGreyscaleBitmap() override;
	virtual void downloadColorBitmap() override;

	virtual void drawOpengl(DataManager& manager) override;
	
private:
	void initGreyscale();
	void initColor();
	void initColorWithInterop();

	void mapColorInteropTexture();
	void unmapColorInteropTexture();

private:
	cudaArray* cuArrayGreyscale = nullptr;
	cudaArray* cuArrayColor = nullptr;
	
	cudaSurfaceObject_t greyscaleSurfObject = 0;
	cudaSurfaceObject_t colorSurfObject = 0;


	bool useOpenGLInterop;

	//interop things
	GLuint colorTexture;
	cudaGraphicsResource_t colorGraphicsResource;

};
