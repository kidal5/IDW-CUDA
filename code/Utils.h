#pragma once

#include <string>

#include "P2.h"

class CpuIdwBase;
class DataManager;

/// Utility class mostly for drawing gui and strings into image
class Utils {
public:
	/// Draw Gui into image
	static void drawGui(DataManager& data, CpuIdwBase& base);

	/// Draw Gui into command line
	static void printGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName);

	/// Draw Gui into image
	static void drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint8_t* img, const P2& imgSize, uint8_t color);

	/// Draw string into image
	static void drawString(const std::string& str, P2 point, uint8_t* img, const P2& imgSize, uint8_t color);

	/// Draw character into image
	static void drawChar(char c, const P2& point, uint8_t* img, const P2& imgSize, uint8_t color);

	/// Draw Gui into image
	static void drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);

	/// Draw string into image
	static void drawString(const std::string& str, P2 point, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);

	/// Draw character into image
	static void drawChar(char c, const P2& point, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);
	
	/// Returns number that is higher power of two
	static int getBiggerPowerOfTwo(int num);

	/// Set vsync refresh interval
	static void setVSync(int interval);

	/// Change vsync state
	static void toggleVsync();

};
