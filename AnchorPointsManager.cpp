#include "AnchorPointsManager.h"

#include <fmt/core.h>
#include <algorithm>
#include <GL/glut.h>

void AnchorPointsManager::handleKeys(const unsigned char key, int x, int y) {
	switch (key) {
	case 27:	// ESC
		exit(0);
	case 9:	// TAB
		idwSelector = (idwSelector + 1) % 3;
		change = true;
		break;
	default:
		break;
	}
}

void AnchorPointsManager::handleSpecialKeys(const int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_UP:
		pParam += 0.1;
		change = true;
		break;
	case GLUT_KEY_DOWN:
		pParam -= 0.1;
		change = true;
		break;
	}
	
}

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

std::vector<P2>& AnchorPointsManager::getAnchorPoints(){
	return anchorPoints;
}

bool AnchorPointsManager::getChange() const {
	return change;
}


void AnchorPointsManager::setChangeDone() {
	change = false;
}

int AnchorPointsManager::getSelectedIdwIndex() const {
	return idwSelector;
}

uint8_t AnchorPointsManager::getMouseValue() const {
	return mouse;
}

double AnchorPointsManager::getPParam() const {
	return pParam;
}

void AnchorPointsManager::handleMouseWheel(const int button, const int state, const int x, const int y) {
	mouse += (button == 3) ? 6 : -6;
	change = true;
}

void AnchorPointsManager::handleLeftButton(const int button, const int state, const int x, const int y) {
	anchorPoints.emplace_back(x, y, mouse);
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
