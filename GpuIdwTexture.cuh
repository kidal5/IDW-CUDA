#pragma once

#include "GpuIdwBase.cuh"
#include "cuda_runtime.h"
//#include "cuda_texture_types.h"
//#include "device_launch_parameters.h"

class GpuIdwTexture final : public GpuIdwBase {

public:
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
	void initBasic();
	void initWithInterop();

	
private:
	cudaArray* cuArrayGreyscale = nullptr;
	cudaArray* cuArrayColor = nullptr;
	
	cudaSurfaceObject_t greyscaleSurfObject = 0;
	cudaSurfaceObject_t colorSurfObject = 0;

	bool useOpenGLInterop;

	//interop things
	GLuint viewGLTexture;
	cudaGraphicsResource_t viewCudaResource;

};
