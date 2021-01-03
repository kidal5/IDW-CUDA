#include "IdwBase.h"


#include <chrono>
#include <utility>

#include <fmt/core.h>

IdwBase::IdwBase(const int _width, const int _height, std::string _methodName)
: width(_width), height(_height), methodName(std::move(_methodName)) {
}

int IdwBase::getWidth() const {
	return static_cast<int>(width);
}

int IdwBase::getHeight() const {
	return static_cast<int>(height);
}

void IdwBase::refresh(AnchorPointsManager& manager) const {
	if (!manager.getChange()) return;

	long long elapsed;
	refresh(manager, elapsed);
	fmt::print("Time for {:<15} is {:8} milliseconds / {:5f} FPS\n", methodName, elapsed, 1.0 / elapsed);
}


void IdwBase::refresh(AnchorPointsManager& manager, long long& elapsedMilliseconds) const {
	if (!manager.getChange()) return;

	const auto timeBegin = std::chrono::system_clock::now();

	refreshInner(manager.getAnchorPoints());
	refreshInnerDrawAnchorPoints(manager.getAnchorPoints());

	elapsedMilliseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - timeBegin).count();
}
