#include "GpuIdwTexture.cuh"

#include "cuda_runtime_api.h"
#include "surface_functions.h"
#include "surface_indirect_functions.h"


static void handleCudaError(const cudaError_t error, const char* file, const int line) {
	if (error == cudaSuccess) return;

	fmt::print("{} in {} at line {}\n", cudaGetErrorString(error), file, line);
	exit(EXIT_FAILURE);
}

#define CHECK_ERROR( error ) ( handleCudaError( error, __FILE__, __LINE__ ) )

namespace
{
	__global__ void gpuDrawAnchorPointsKernel(cudaSurfaceObject_t surfObject, const int* anchorPoints, const int anchorPointsCount, const int width, const int height) {

		const int x = threadIdx.x;

		if (x < anchorPointsCount) {
			const int xAnchor = anchorPoints[3 * x];
			const int yAnchor = anchorPoints[3 * x + 1];

			uchar1 data;
			surf2Dread(&data, surfObject, xAnchor + 1, yAnchor);
			data.x = data.x > 127 ? 0 : 255;

			for (int shiftX = -1; shiftX < 1; shiftX++) {
				for (int shiftY = -1; shiftY < 1; shiftY++) {
					surf2Dwrite(data, surfObject, xAnchor + shiftX , yAnchor + shiftY);
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
}

GpuIdwTexture::GpuIdwTexture(const int _width, const int _height) : GpuIdwBase(_width, _height, "GpuIdwTexture") {

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	CHECK_ERROR(cudaMallocArray(&cuArrayGreyscale, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	// Specify surface
	struct cudaResourceDesc resDesc{};
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	// Create the surface object
	resDesc.res.array.array = cuArrayGreyscale;
	cudaCreateSurfaceObject(&surfObject, &resDesc);

}

GpuIdwTexture::~GpuIdwTexture() {

	if (surfObject)
		CHECK_ERROR(cudaDestroySurfaceObject(surfObject));
	
	if (cuArrayGreyscale)
		CHECK_ERROR(cudaFreeArray(cuArrayGreyscale));

}

uint8_t* GpuIdwTexture::getBitmapGreyscaleCpu() {

	if (!lastVersionOnCpu) {
		CHECK_ERROR(cudaMemcpyFromArray(bitmapGreyscaleCpu, cuArrayGreyscale, 0, 0, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		lastVersionOnCpu = true;
	}

	return bitmapGreyscaleCpu;
}

void GpuIdwTexture::refreshInnerGpu(const double pParam) {

	dim3 gridRes(width / 32, height / 32);
	dim3 blockRes(32, 32);

	//// Invoke kernel
	//dim3 dimBlock(16, 16);
	//dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
	//	(height + dimBlock.y - 1) / dimBlock.y);

	gpuTextureKernel << < gridRes, blockRes >> > (surfObject, anchorsGpu, anchorsGpuCurrentCount, pParam, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

void GpuIdwTexture::refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) {

	int power = 1;
	while (power < anchorsGpuCurrentCount)
		power *= 2;

	if (power >= 1024) {
		throw std::exception("power is bigger than 1024");
	}


	gpuDrawAnchorPointsKernel << < 1, power >> > (surfObject, anchorsGpu, anchorsGpuCurrentCount, width, height);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

