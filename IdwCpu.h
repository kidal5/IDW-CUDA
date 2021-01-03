#pragma once

#include "IdwBase.h"


class IdwCpu final : public IdwBase{
public:
	IdwCpu(int _width, int _height);
	
	virtual ~IdwCpu() override = default;
	virtual void* getBitmapCpu() override;
private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints) const override;
	virtual void refreshInnerDrawAnchorPoints(const std::vector<P2>& anchorPoints) const override;

private:
	std::unique_ptr<uint8_t[]> bitmap;

};
