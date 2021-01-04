#include "CpuIdwThreaded.h"

#include <thread>


namespace 
{
	void threadJob(
		const int xStart, const int yStart, const int xChunkSize, const int yChunkSize, 
		const int width, const int height, const double pParam, const std::vector<P2>& anchorPoints, uint8_t* data)
	{
		//fill its own chunk
		for (int h = yStart; h < yStart + yChunkSize; ++h) {
			for (int w = xStart; w < xStart + xChunkSize; ++w) {
				double wiSum = 0;
				double outputSum = 0;

				for (const auto& point : anchorPoints) {

					const double wi = CpuIdwBase::computeWiCpu({ w,h, 0 }, point, pParam);
					wiSum += wi;
					outputSum += wi * point.value;
				}
				outputSum /= wiSum;
				
				data[4 * (h * width + w) + 0] = static_cast<uint8_t>(outputSum);
				data[4 * (h * width + w) + 1] = static_cast<uint8_t>(outputSum);
				data[4 * (h * width + w) + 2] = static_cast<uint8_t>(outputSum);
				data[4 * (h * width + w) + 4] = static_cast<uint8_t>(outputSum);
			}
		}
	}
}

CpuIdwThreaded::CpuIdwThreaded(const int _width, const int _height, const int _numOfThreads)
	: CpuIdwBase(_width, _height, "CpuIdwThreaded"), numOfThreads(_numOfThreads) {

	const auto processorCount = std::thread::hardware_concurrency();

	if (processorCount < numOfThreads || numOfThreads == 1) {
		numOfThreads = processorCount;
	}
	
}

void CpuIdwThreaded::refreshInner(const std::vector<P2>& anchorPoints, double pParam) {

	const int xChunkSizeDefault = 256;
	const int yChunkSizeDefault = 128;

	uint8_t * p = bitmapCpu.get();

	//split image into 16x16 chunks and let every thread do it's own work

	std::vector<std::thread> threads(numOfThreads);
	int currentThreadId = 0;
	
	for (int h = 0; h < height; h = h + yChunkSizeDefault) {
		for (int w = 0; w < width; w = w + xChunkSizeDefault) {

			// check chunk size
			int xChunkSize = (w + xChunkSizeDefault > width) ? (width - w) : xChunkSizeDefault;
			int yChunkSize = (h + yChunkSizeDefault > height) ? (height - h) : yChunkSizeDefault;

			if (threads[currentThreadId].joinable())
				threads[currentThreadId].join();

			threads[currentThreadId] = std::thread(threadJob, w, h, xChunkSize, yChunkSize, width, height, pParam, anchorPoints, p);
			currentThreadId = (currentThreadId + 1) % numOfThreads;
		}
	}

	for (auto & thread : threads) {
		if (thread.joinable()) 
			thread.join();
	}
	
}

