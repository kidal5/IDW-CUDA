#pragma once

#include <string>

#include "P2.h"

class CpuIdwBase;
class DataManager;

class Utils {
public:
	static void drawGui(DataManager& data, CpuIdwBase& base);

	static void printGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName);

	static void drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint8_t* img, const P2& imgSize, uint8_t color);
	static void drawString(const std::string& str, P2 point, uint8_t* img, const P2& imgSize, uint8_t color);
	static void drawChar(char c, const P2& point, uint8_t* img, const P2& imgSize, uint8_t color);

	static void drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);
	static void drawString(const std::string& str, P2 point, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);
	static void drawChar(char c, const P2& point, uint32_t* img, const P2& imgSize, uint32_t rgbaColor);

	static int getBiggerPowerOfTwo(int num);

	static void setVSync(int interval);
	static void toggleVsync();

};
