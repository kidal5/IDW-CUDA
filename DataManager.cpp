#include "DataManager.h"

#include <fmt/core.h>
#include <fmt/color.h>
#include <algorithm>
#include <GL/glut.h>

#include <iostream>
#include <fstream>


void DataManager::handleKeys(const unsigned char key, int x, int y) {
	switch (key) {
	case 27:	// ESC
		exit(0);
	case 9:	// TAB
		idwSelector = (idwSelector + 1) % idwSelectorModulo;
		change = true;
		break;
	case '1':
		readDataFromFile("xd");
		break;
	case '0':
		dumpDataToFile();
		break;

	case 'c':
		anchorPoints.clear();
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

	const auto elapsedMs= std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - fpsCounterStart).count();

	lastFps = 1000000.0 / elapsedMs;
		
	fpsCounterStart = std::chrono::high_resolution_clock::now();

	//if (elapsedMs >= 100000) {
	//	fpsCounterStart = std::chrono::high_resolution_clock::now();
	//	last
	//	lastFps = fpsCounter * 10;
	//	fpsCounter = 0;
	//}

	//fpsCounter++;
}

void DataManager::setNumberOfIdws(const int number) {
	idwSelectorModulo = number;
}

int DataManager::getFullFPS() const {
	return lastFps;
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

void DataManager::dumpDataToFile() {

	std::ofstream f("data.bin", std::ios::out | std::ios::binary);
	if (f.is_open()) {
		const auto* rawPointer = reinterpret_cast<const char*>(anchorPoints.data());
		f.write(rawPointer, sizeof(P2) * anchorPoints.size());
	} else {
		fmt::print(fg(fmt::color::red), "Unable to write to file.\n");
	}
}

void DataManager::readDataFromFile(std::string fname) {


	std::ifstream file("data.bin", std::ios::in | std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		fmt::print(fg(fmt::color::red), "Unable to read from file. / {}\n", fname);
		return;
	}
	
	auto size = static_cast<long long>(file.tellg());
	size /= sizeof(P2);

	anchorPoints.clear();
	anchorPoints.reserve(size);
	for (int i = 0; i < size; ++i) {
		// "create x objects in vector"
		anchorPoints.emplace_back(0, 0, 0);
	}
	
	auto* rawPointer = reinterpret_cast<char*>(anchorPoints.data());

	file.seekg(0, std::ios::beg);
	file.read(rawPointer, sizeof(P2) * size);
	change = true;
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
