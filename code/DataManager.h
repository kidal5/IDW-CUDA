#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <GL/glut.h>

#include "ColorPalette.h"

#include "P2.h"

struct Palette{
	std::string name;
	GLenum drawFormat;
	bool isEightBit;
	uint32_t* colorData = nullptr;
};

class DataManager {
public:

	DataManager();

	void handleKeys(unsigned char key, int x, int y);
	void handleSpecialKeys(int key, int x, int y);
	void handleMouse(int button, int state, int x, int y);
	std::vector<P2>& getAnchorPoints();
	bool getChange() const;
	void setChangeDone();
	void setNumberOfIdws(int number);
	int getFullFPS() const;

	int getCurrentIdw() const;
	Palette getCurrentPalette() const;

	uint8_t getMouseValue() const;
	double getPParam() const;

	void dumpDataToFile();
	void readDataFromFile(std::string fname);
	
private:
	void handleMouseWheel(const int button, const int state, const int x, const int y);
	void handleLeftButton(const int button, const int state, const int x, const int y);
	void handleRightButton(const int button, const int state, const int x, const int y);
	void generateRandomPoints();


private:
	bool change = true;
	uint8_t mouse = 255;
	double pParam = 4.2;
	int idwSelector = 0;
	int idwSelectorModulo = -1;
	int paletteIndex = 3;


	std::chrono::steady_clock::time_point fpsCounterStart = std::chrono::high_resolution_clock::now();
	int fpsCounter = 0;
	int lastFps = 0;


	std::vector<Palette> palettes = {
		{"greyscale", GL_LUMINANCE, true},
		{"red", GL_RED, true},
		{"blue", GL_BLUE, true},
		{"green", GL_GREEN, true},
		{"inferno", GL_RGBA, false, ColorPalette::inferno},
		{"magma", GL_RGBA, false, ColorPalette::magma},
		{"moreland", GL_RGBA, false, ColorPalette::moreland},
		{"plasma", GL_RGBA, false, ColorPalette::plasma},
		{"viridis", GL_RGBA, false, ColorPalette::viridis},
	};

	std::vector<P2> anchorPoints;
	//std::vector<P2> anchorPoints = { {100, 10, 10}, {100, 50, 50}, {100, 100, 100}, {150, 150, 150}, {200, 200, 200}, {255, 255, 255} };

};
