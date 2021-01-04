#include "GpuIdwBase.cuh"

#include "Utils.h"

static void handleCudaError(const cudaError_t error, const char* file, const int line) {
	if (error == cudaSuccess) return;

	fmt::print("{} in {} at line {}\n", cudaGetErrorString(error), file, line);
	exit(EXIT_FAILURE);
}

#define CHECK_ERROR( error ) ( handleCudaError( error, __FILE__, __LINE__ ) )


GpuIdwBase::GpuIdwBase(const int _width, const int _height, const std::string& _methodName) : CpuIdwBase(_width, _height, _methodName) {

	CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&anchorsGpu), anchorsGpuBytes));
	CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&colorMappingData), 256 * sizeof(uchar4)));

	blockRes = dim3(16, 16);
	gridRes = dim3((width + blockRes.x - 1) / blockRes.x, (height + blockRes.y - 1) / blockRes.y);
}

GpuIdwBase::~GpuIdwBase() {

	if (anchorsGpu) 
		CHECK_ERROR(cudaFree(anchorsGpu));

	if (colorMappingData)
		CHECK_ERROR(cudaFree(colorMappingData));
}

uint8_t* GpuIdwBase::getBitmapGreyscaleCpu() {

	if (!lastGreyscaleVersionOnCpu){
		downloadGreyscaleBitmap();
		lastGreyscaleVersionOnCpu = true;
	}

	return bitmapGreyscaleCpu;
}

uint32_t* GpuIdwBase::getBitmapColorCpu() {
	if (!lastColorVersionOnCpu) {
		downloadColorBitmap();
		lastColorVersionOnCpu = true;
	}

	return bitmapColorCpu;
}


void GpuIdwBase::refreshInnerGreyscale(DataManager& manager) {
	lastGreyscaleVersionOnCpu = false;
	copyAnchorsToGpu(manager.getAnchorPoints());
	refreshInnerGreyscaleGpu(manager.getPParam());
}

void GpuIdwBase::refreshInnerColor(const Palette& p) {
	lastColorVersionOnCpu = false;
	copyPaletteToGpu(p);
	refreshInnerColorGpu();
}

void GpuIdwBase::copyAnchorsToGpu(const std::vector<P2>& anchorPoints) {

	if (anchorPoints.size() > anchorsGpuMaxCount) {
		//free memory
		if (anchorsGpu) CHECK_ERROR(cudaFree(anchorsGpu));

		anchorsGpuMaxCount = anchorsGpuMaxCount * 2;
		anchorsGpuBytes = anchorsGpuMaxCount * 3 * sizeof(int);

		//create bigger memory
		CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&anchorsGpu), anchorsGpuBytes));
	}

	anchorsGpuCurrentCount = anchorPoints.size();
	gridAnchorRes = dim3((anchorPoints.size() + blockAnchorRes.x - 1) / blockAnchorRes.x, 1);

	//i should be able to just read vector's data as ints ...

	const auto* rawPointer = reinterpret_cast<const int*>(anchorPoints.data());

	CHECK_ERROR(cudaMemcpy(anchorsGpu, rawPointer, anchorsGpuBytes, cudaMemcpyHostToDevice));
}

void GpuIdwBase::copyPaletteToGpu(const Palette& p) {

	if (p.name == lastPaletteName) return;

	lastPaletteName = p.name;
	CHECK_ERROR(cudaMemcpy(colorMappingData, p.colorData, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));
}
