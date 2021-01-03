#pragma once

#include "IdwBase.h"


class IdwCpu final : public IdwBase{
public:
	IdwCpu(const int _width, const int _height) : IdwBase(_width, _height, "IdwCpu") {}
	virtual ~IdwCpu() override = default;
private:
	virtual void refreshInner(const std::vector<P2>& anchorPoints) override;
};
