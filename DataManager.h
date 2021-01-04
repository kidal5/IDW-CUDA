#pragma once
#include <string>
#include <vector>
#include <GL/glut.h>

#include "P2.h"

struct Palette{
	std::string name;
	GLenum drawFormat;
};

class DataManager {
public:

	void handleKeys(unsigned char key, int x, int y);
	void handleSpecialKeys(int key, int x, int y);
	void handleMouse(int button, int state, int x, int y);
	std::vector<P2>& getAnchorPoints();
	bool getChange() const;
	void setChangeDone();

	int getCurrentIdw() const;
	Palette getCurrentPalette() const;

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
	int paletteIndex = 0;

	std::vector<Palette> palettes = {
		{"greyscale", GL_LUMINANCE},
		{"red", GL_RED}
	};

	std::vector<P2> anchorPoints = {{100, 10, 10}, {100, 50, 50}, {100, 100, 100}, {150, 150, 150}, {200, 200, 200}, {255, 255, 255}};

};
