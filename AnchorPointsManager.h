#pragma once
#include <vector>

#include "P2.h"

class AnchorPointsManager {
public:

	void handleKeys(unsigned char key, int x, int y);
	void handleSpecialKeys(int key, int x, int y);
	void handleMouse(int button, int state, int x, int y);
	std::vector<P2>& getAnchorPoints();
	bool getChange() const;
	void setChangeDone();

	int getSelectedIdwIndex() const;

	uint8_t getMouseValue() const;
	double getPParam() const;

private:
	void handleMouseWheel(const int button, const int state, const int x, const int y);
	void handleLeftButton(const int button, const int state, const int x, const int y);
	void handleRightButton(const int button, const int state, const int x, const int y);
	

private:
	bool change = true;
	uint8_t mouse = 255;
	double pParam = 4.2;
	int idwSelector = 0;

	std::vector<P2> anchorPoints = {{100, 10, 10}, {100, 50, 50}, {100, 100, 100}, {150, 150, 150}, {200, 200, 200}, {255, 255, 255}};

};
