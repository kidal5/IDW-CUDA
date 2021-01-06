#pragma once

//https://stackoverflow.com/a/46991956
///Helper class to change between fullscreen and windowed mode
class FullscreenHelper {
public:

	///Respond to key f
	static bool handleKeys(const unsigned char key, const int x, const int y);

private:
	static void setWindowed(int positionX, int positionY, int width, int height);

	static int isFullScreen;
	static int previousPosition[2];
	static int previousSize[2];
};

