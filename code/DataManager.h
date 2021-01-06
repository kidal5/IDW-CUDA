#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <GL/glut.h>

#include "ColorPalette.h"

#include "P2.h"

/// Palette defines mapping from greyscale to color image
struct Palette{
	std::string name;
	GLenum drawFormat;
	bool isEightBit;
	uint32_t* colorData = nullptr;
};

/// Holds state of running application
class DataManager {
public:

	DataManager();

	/// respond to user pressed keys
	void handleKeys(unsigned char key, int x, int y);

	/// respond to user pressed keys
	void handleSpecialKeys(int key, int x, int y);

	/// respond to user mouse movement
	void handleMouse(int button, int state, int x, int y);

	/// returns anchor points
	std::vector<P2>& getAnchorPoints();

	/// returns true when some change has occured since last refresh
	bool getChange() const;

	/// signalize, that refresh has been completed
	void setChangeDone();

	/// set how many idw methods is currenlty available
	void setNumberOfIdws(int number);

	/// get fps 
	int getFullFPS() const;

	/// get current idw
	int getCurrentIdw() const;

	/// get current palette
	Palette getCurrentPalette() const;

	/// get mouse value, in range 0 -> 256
	uint8_t getMouseValue() const;
	
	/// get p param for idw computing
	double getPParam() const;

	/// dump all anchor points into user file
	void dumpDataToFile();
	
	/// read data from file
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
