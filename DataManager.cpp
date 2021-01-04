#include "DataManager.h"

#include <fmt/core.h>
#include <algorithm>
#include <GL/glut.h>

void DataManager::handleKeys(const unsigned char key, int x, int y) {
	switch (key) {
	case 27:	// ESC
		exit(0);
	case 9:	// TAB
		idwSelector = (idwSelector + 1) % idwSelectorModulo;
		change = true;
		break;
	default:
		break;
	}
}

void DataManager::handleSpecialKeys(const int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_UP:
		pParam += 0.1;
		change = true;
		break;
	case GLUT_KEY_DOWN:
		pParam -= 0.1;
		change = true;
		break;
	case GLUT_KEY_LEFT:
		paletteIndex = (paletteIndex + palettes.size() - 1) % palettes.size();
		change = true;
		break;
	case GLUT_KEY_RIGHT:
		paletteIndex = (paletteIndex + 1) % palettes.size();
		change = true;
		break;
	}
	
}

void DataManager::handleMouse(const int button, const int state, const int x, const int y) {

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

std::vector<P2>& DataManager::getAnchorPoints(){
	return anchorPoints;
}

bool DataManager::getChange() const {
	return change;
}


void DataManager::setChangeDone() {
	change = false;
}

void DataManager::setNumberOfIdws(const int number) {
	idwSelectorModulo = number;
}

int DataManager::getCurrentIdw() const {
	return idwSelector;
}

Palette DataManager::getCurrentPalette() const {
	return palettes[paletteIndex];
}


uint8_t DataManager::getMouseValue() const {
	return mouse;
}

double DataManager::getPParam() const {
	return pParam;
}

void DataManager::handleMouseWheel(const int button, const int state, const int x, const int y) {
	mouse += (button == 3) ? 6 : -6;
	change = true;
}

void DataManager::handleLeftButton(const int button, const int state, const int x, const int y) {
	anchorPoints.emplace_back(x, y, mouse);
	change = true;
}

void DataManager::handleRightButton(const int button, const int state, const int x, const int y) {

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
