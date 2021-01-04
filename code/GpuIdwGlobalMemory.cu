#include "GpuIdwGlobalMemory.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void handleCudaError(const cudaError_t error, const char* file, const int line) {
	if (error == cudaSuccess) return;

	fmt::print("{} in {} at line {}\n", cudaGetErrorString(error), file, line);
	exit(EXIT_FAILURE);
}

#define CHECK_ERROR( error ) ( handleCudaError( error, __FILE__, __LINE__ ) )

namespace
{
	__global__ void gpuDrawAnchorPointsKernel(uint8_t* bitmap, const int* anchorPoints, const int anchorPointsCount, const int width, const int height) {

		const int x = threadIdx.x;

		if (x < anchorPointsCount) {
			const int xAnchor = anchorPoints[3 * x];
			const int yAnchor = anchorPoints[3 * x + 1];

			uint8_t value = bitmap[yAnchor * width + xAnchor + 1] > 127 ? 0 : 255;

			for (int shiftX = -1; shiftX < 1; shiftX++) {
				for (int shiftY = -1; shiftY < 1; shiftY++) {

					bitmap[(yAnchor + shiftY) * width + xAnchor + shiftX] = value;
				}
			}
		}
	}
	
	__device__ double computeWiGpu(const int ax, const int ay, const int bx, const int by, const double pParam) {
		const float dist = sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
		return 1 / powf(dist, pParam);
	}

	__global__ void gpuGlobalMemoryKernel(uint8_t* bitmap, const int* anchorPoints, const int anchorPointsCount, const double pParam, const int width, const int height) {

		const int x= blockIdx.x * blockDim.x + threadIdx.x;
		const int y= blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			double wiSum = 0;
			double outputSum = 0;

			for (int i = 0; i < anchorPointsCount; i++) {
				const double wi = computeWiGpu(x, y, anchorPoints[i * 3], anchorPoints[i * 3 + 1], pParam);
				wiSum += wi;
				outputSum += wi * anchorPoints[i * 3 + 2];
			}
			outputSum /= wiSum;

			bitmap[y * width + x] = static_cast<uint8_t>(outputSum);
			//bitmap[4 * (y * width + x) + 0] = static_cast<uint8_t>(outputSum);
		}
	}

	__global__ void gpuGlobalMemoryColorKernel(const uint8_t* input, uchar4* output, uchar4 * colorData, const int width, const int height) {

		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			const uint8_t val = input[y * width + x];
			output[y * width + x] = colorData[val];
		}
	}
}


GpuIdwGlobalMemory::GpuIdwGlobalMemory(const int _width, const int _height) : GpuIdwBase(_width, _height, "GpuIdwGlobalMemory") {

	bitmapGreyscaleBytesCount = width * height * sizeof(uint8_t);
	bitmapColorBytesCount = width * height * sizeof(uchar4);

	CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&bitmapGreyscaleGpu), bitmapGreyscaleBytesCount));
	CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&bitmapColorGpu), bitmapColorBytesCount));

}

GpuIdwGlobalMemory::~GpuIdwGlobalMemory() {

	if (bitmapGreyscaleGpu)
		CHECK_ERROR(cudaFree(bitmapGreyscaleGpu));

	if (bitmapColorGpu)
		CHECK_ERROR(cudaFree(bitmapColorGpu));
}

void GpuIdwGlobalMemory::refreshInnerGreyscaleGpu(const double pParam) {

	gpuGlobalMemoryKernel <<< gridRes, blockRes>>>(bitmapGreyscaleGpu, anchorsGpu, anchorsGpuCurrentCount, pParam, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

void GpuIdwGlobalMemory::refreshInnerGreyscaleDrawAnchorPoints(const std::vector<P2>& anchorPoints) {

	gpuDrawAnchorPointsKernel<< < gridAnchorRes, blockAnchorRes>> > (bitmapGreyscaleGpu, anchorsGpu, anchorsGpuCurrentCount, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

void GpuIdwGlobalMemory::refreshInnerColorGpu() {

	gpuGlobalMemoryColorKernel<< < gridRes, blockRes >> > (bitmapGreyscaleGpu, bitmapColorGpu, colorMappingData, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

void GpuIdwGlobalMemory::downloadGreyscaleBitmap() {
	CHECK_ERROR(cudaMemcpy(bitmapGreyscaleCpu, bitmapGreyscaleGpu, bitmapGreyscaleBytesCount, cudaMemcpyDeviceToHost));
}

void GpuIdwGlobalMemory::downloadColorBitmap() {
	CHECK_ERROR(cudaMemcpy(bitmapColorCpu, bitmapColorGpu, bitmapColorBytesCount, cudaMemcpyDeviceToHost));
}
