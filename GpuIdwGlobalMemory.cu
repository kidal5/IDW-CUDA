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

			bitmap[4 * (y * width + x) + 0] = static_cast<uint8_t>(outputSum);
			bitmap[4 * (y * width + x) + 1] = static_cast<uint8_t>(outputSum);
			bitmap[4 * (y * width + x) + 2] = static_cast<uint8_t>(outputSum);
			bitmap[4 * (y * width + x) + 3] = static_cast<uint8_t>(outputSum);
		}
	}
}


GpuIdwGlobalMemory::GpuIdwGlobalMemory(const int _width, const int _height) : GpuIdwBase(_width, _height, "GpuIdwGlobalMemory") {

	imgBytesCount = width * height * 4 * sizeof(uint8_t);

	CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&bitmapGpu), imgBytesCount));
	
}

GpuIdwGlobalMemory::~GpuIdwGlobalMemory() {

	if (bitmapGpu)
		CHECK_ERROR(cudaFree(bitmapGpu));
	
}

uint8_t* GpuIdwGlobalMemory::getBitmapCpu() {

	if (!lastVersionOnCpu) {
		CHECK_ERROR(cudaMemcpy(bitmapCpu.get(), bitmapGpu, imgBytesCount, cudaMemcpyDeviceToHost));

		lastVersionOnCpu = true;
	}

	return bitmapCpu.get();
}

void GpuIdwGlobalMemory::refreshInnerGpu(const double pParam) {

	dim3 gridRes(width / 32, height / 32);
	dim3 blockRes(32, 32);

	gpuGlobalMemoryKernel <<< gridRes, blockRes>>>(bitmapGpu, anchorsGpu, anchorsGpuCurrentCount, pParam, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}
