#pragma once

#include <memory>
#include <vector>

#include "AnchorPointsManager.h"
#include "P2.h"

#include <string>

class IdwBase {
public:
	virtual ~IdwBase() = default;
	IdwBase(int _width, int _height, std::string _methodName);

	int getWidth() const;
	int getHeight() const;

	void refresh(AnchorPointsManager& manager) const;
	void refresh(AnchorPointsManager& manager, long long& elapsedMilliseconds) const;
	virtual void* getBitmapCpu() = 0;

private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints) const = 0;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) const = 0;

	
protected:
	long long width;
	long long height;
	std::string methodName;
	
};

