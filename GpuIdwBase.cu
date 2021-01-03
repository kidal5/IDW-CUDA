#include "GpuIdwBase.cuh"

static void handleCudaError(const cudaError_t error, const char* file, const int line) {
	if (error == cudaSuccess) return;

	fmt::print("{} in {} at line {}\n", cudaGetErrorString(error), file, line);
	exit(EXIT_FAILURE);
}

#define CHECK_ERROR( error ) ( handleCudaError( error, __FILE__, __LINE__ ) )


GpuIdwBase::GpuIdwBase(const int _width, const int _height, const std::string& _methodName) : IdwBase(_width, _height, _methodName) {

	imgBytesCount = width * height * 3 * sizeof(uint8_t);

	auto err = cudaMalloc(reinterpret_cast<void**>(&bitmapGpu), imgBytesCount);
	CHECK_ERROR(err);

	err = cudaMalloc(reinterpret_cast<void**>(&anchorsGpu), anchorsGpuBytes);
	CHECK_ERROR(err);
}

GpuIdwBase::~GpuIdwBase() {

	if (bitmapGpu)
		CHECK_ERROR(cudaFree(bitmapGpu));

	if (anchorsGpu)
		CHECK_ERROR(cudaFree(anchorsGpu));

}

uint8_t* GpuIdwBase::getBitmapCpu() {

	if (!lastVersionOnCpu) {
		auto err = cudaMemcpy(bitmapCpu.get(), bitmapGpu, imgBytesCount, cudaMemcpyDeviceToHost);

		lastVersionOnCpu = true;
	}

	return bitmapCpu.get();
}

void GpuIdwBase::refreshInner(const std::vector<P2>& anchorPoints, const double pParam) {
	lastVersionOnCpu = false;
	copyAnchorsToGpu(anchorPoints);
	refreshInnerGpu(pParam);
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

	//i should be able to just read vector's data as ints ...

	const auto* rawPointer = reinterpret_cast<const int*>(anchorPoints.data());

	const auto err = cudaMemcpy(anchorsGpu, rawPointer, anchorsGpuBytes, cudaMemcpyHostToDevice);
	CHECK_ERROR(err);
}

