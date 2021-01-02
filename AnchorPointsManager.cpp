#include "AnchorPointsManager.h"



#include <algorithm>
#include <GL/glut.h>

void AnchorPointsManager::handleMouse(const int button, const int state, const int x, const int y) {

	if (state == GLUT_DOWN) return;

    if (button == 3 || button == 4) {
        handleMouseWheel(button, state, x, y);
        return;
    }

	if (button == GLUT_LEFT_BUTTON) {
		handleLeftButton(button, state, x, y);
		return;
	}

	if (button == GLUT_RIGHT_BUTTON) {
		handleRightButton(button, state, x, y);
		return;
	}
}

std::vector<P2>& AnchorPointsManager::getAnchorPoints(bool& changeFromLastCall) {
	changeFromLastCall = change;
	change = false;

	return anchorPoints;
}

void AnchorPointsManager::handleMouseWheel(const int button, const int state, const int x, const int y) {

	mouse += (button == 3) ? 10 : -10;
	LOG("Mouse: " + std::to_string(static_cast<int>(mouse)));
}

void AnchorPointsManager::handleLeftButton(const int button, const int state, const int x, const int y) {
	anchorPoints.push_back({ x, y, mouse });
	change = true;
}

void AnchorPointsManager::handleRightButton(const int button, const int state, const int x, const int y) {

	P2 target = {x,y};
	
	auto pred = [&target](const P2& a, const P2& b) {
		return (a - target).norm2d() < (b - target).norm2d();
	};

	const auto minIndex = std::min_element(anchorPoints.begin(), anchorPoints.end(), pred);

	if ((*minIndex - target).norm2d() < 10) {
		anchorPoints.erase(minIndex);
		change = true;
	}
}
