#include "GpuIdwTexture.cuh"

#include "cuda_runtime_api.h"
#include "surface_functions.h"
#include "surface_indirect_functions.h"

#include <cuda_gl_interop.h>

static void handleCudaError(const cudaError_t error, const char* file, const int line) {
	if (error == cudaSuccess) return;

	fmt::print("{} in {} at line {}\n", cudaGetErrorString(error), file, line);
	exit(EXIT_FAILURE);
}

#define CHECK_ERROR( error ) ( handleCudaError( error, __FILE__, __LINE__ ) )

namespace
{
	__global__ void gpuDrawAnchorPointsKernel(cudaSurfaceObject_t surfObject, const int* anchorPoints, const int anchorPointsCount, const int width, const int height) {

		const int x = blockIdx.x * blockDim.x + threadIdx.x;

		
		if (x < anchorPointsCount) {
			const int xAnchor = anchorPoints[3 * x];
			const int yAnchor = anchorPoints[3 * x + 1];

			uchar1 data;
			surf2Dread(&data, surfObject, xAnchor + 1, yAnchor);
			data.x = data.x > 127 ? 0 : 255;

			for (int shiftX = -1; shiftX < 1; shiftX++) {
				for (int shiftY = -1; shiftY < 1; shiftY++) {
					if (xAnchor + shiftX < width && yAnchor + shiftY < height) {
						surf2Dwrite(data, surfObject, xAnchor + shiftX, yAnchor + shiftY);
					}
				}
			}

		}
	}
	
	__device__ double computeWiGpu(const int ax, const int ay, const int bx, const int by, const double pParam) {
		const float dist = sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
		return 1 / powf(dist, pParam);
	}

	__global__ void gpuTextureKernel(cudaSurfaceObject_t surfObject, const int* anchorPoints, const int anchorPointsCount, const double pParam, const int width, const int height)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (x < width && y < height) {
			double wiSum = 0;
			double outputSum = 0;

			for (int i = 0; i < anchorPointsCount; i++) {
				const double wi = computeWiGpu(x, y, anchorPoints[i * 3], anchorPoints[i * 3 + 1], pParam);
				wiSum += wi;
				outputSum += wi * anchorPoints[i * 3 + 2];
			}
			outputSum /= wiSum;

			
			const auto outputSum8 = static_cast<uint8_t>(outputSum);
			surf2Dwrite(make_uchar1(outputSum8), surfObject, x, y);
		}
	}

	__global__ void gpuTextureColorKernel(const cudaSurfaceObject_t input, cudaSurfaceObject_t output, uchar4* colorData, const int width, const int height) {

		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {

			uchar1 data;
			surf2Dread(&data, input, x, y);
			surf2Dwrite(colorData[data.x], output, x * 4, y);
		}
	}

	__global__ void gpuTextureColorKernelChangeEndian(const cudaSurfaceObject_t input, cudaSurfaceObject_t output, uchar4* colorData, const int width, const int height) {

		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {

			uchar1 data;
			surf2Dread(&data, input, x, y);

			const uchar4 u = colorData[data.x];

			surf2Dwrite(make_uchar4(u.w, u.z, u.y, u.x), output, x * 4, y);
			//surf2Dwrite(colorData[data.x], output, x * 4, y);
		}
	}
}




GpuIdwTexture::GpuIdwTexture(const int _width, const int _height, const bool _useOpenGLInterop)
: GpuIdwBase(_width, _height, _useOpenGLInterop ? "GpuIdwTextureOpenGL" : "GpuIdwTexture"), useOpenGLInterop(_useOpenGLInterop) {

	initGreyscale();
	if (useOpenGLInterop) {
		initColorWithInterop();
	} else {
		initColor();
	}
	
}

GpuIdwTexture::~GpuIdwTexture() {

	if (greyscaleSurfObject && !useOpenGLInterop)
		CHECK_ERROR(cudaDestroySurfaceObject(greyscaleSurfObject));

	if (colorSurfObject && !useOpenGLInterop)
		CHECK_ERROR(cudaDestroySurfaceObject(colorSurfObject));
	
	if (cuArrayGreyscale)
		CHECK_ERROR(cudaFreeArray(cuArrayGreyscale));

	if (cuArrayColor)
		CHECK_ERROR(cudaFreeArray(cuArrayColor));
}

void GpuIdwTexture::refreshInnerGreyscaleGpu(const double pParam) {

	gpuTextureKernel << < gridRes, blockRes >> > (greyscaleSurfObject, anchorsGpu, anchorsGpuCurrentCount, pParam, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

}

void GpuIdwTexture::refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) {

	gpuDrawAnchorPointsKernel << < gridAnchorRes, blockAnchorRes >> > (greyscaleSurfObject, anchorsGpu, anchorsGpuCurrentCount, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}


void GpuIdwTexture::refreshInnerColorGpu() {

	if (useOpenGLInterop)  mapColorInteropTexture();

	if (useOpenGLInterop) {
		gpuTextureColorKernelChangeEndian << < gridRes, blockRes >> > (greyscaleSurfObject, colorSurfObject, colorMappingData, width, height);
	} else {
		gpuTextureColorKernel << < gridRes, blockRes >> > (greyscaleSurfObject, colorSurfObject, colorMappingData, width, height);
	}
	
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	if (useOpenGLInterop)  unmapColorInteropTexture();
	
}

void GpuIdwTexture::downloadGreyscaleBitmap() {
	CHECK_ERROR(cudaMemcpyFromArray(bitmapGreyscaleCpu, cuArrayGreyscale, 0, 0, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void GpuIdwTexture::downloadColorBitmap() {

	if (useOpenGLInterop) throw std::exception("Color bitmap download is disabled in OpenGL interop mode.");
	
	CHECK_ERROR(cudaMemcpyFromArray(bitmapColorCpu, cuArrayColor, 0, 0, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
}

void GpuIdwTexture::drawOpengl(DataManager& manager) {

	if (!useOpenGLInterop || manager.getCurrentPalette().isEightBit) {
		GpuIdwBase::drawOpengl(manager);
		return;
	}

	glBindTexture(GL_TEXTURE_2D, colorTexture);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glFinish();
}

void GpuIdwTexture::initGreyscale() {
	//greyscale
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	CHECK_ERROR(cudaMallocArray(&cuArrayGreyscale, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	// Specify surface
	struct cudaResourceDesc resDesc {};
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArrayGreyscale;
	CHECK_ERROR(cudaCreateSurfaceObject(&greyscaleSurfObject, &resDesc));
}

void GpuIdwTexture::initColor() {
	//color
	auto channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	CHECK_ERROR(cudaMallocArray(&cuArrayColor, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	struct cudaResourceDesc resDesc {};
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArrayColor;
	CHECK_ERROR(cudaCreateSurfaceObject(&colorSurfObject, &resDesc));
}

void GpuIdwTexture::initColorWithInterop() {

	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, &colorTexture);
	glBindTexture(GL_TEXTURE_2D, colorTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	CHECK_ERROR(cudaGraphicsGLRegisterImage(&colorGraphicsResource, colorTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

}

void GpuIdwTexture::mapColorInteropTexture() {

	//texture has to be mapped every time it's used by cuda and then unmapped
	
	CHECK_ERROR(cudaGraphicsMapResources(1, &colorGraphicsResource));
	cudaArray_t viewCudaArray;
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, colorGraphicsResource, 0, 0));
	cudaResourceDesc viewCudaArrayResourceDesc;
	viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
	viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

	CHECK_ERROR(cudaCreateSurfaceObject(&colorSurfObject, &viewCudaArrayResourceDesc));
}

void GpuIdwTexture::unmapColorInteropTexture() {
	CHECK_ERROR(cudaDestroySurfaceObject(colorSurfObject));
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &colorGraphicsResource));
}


