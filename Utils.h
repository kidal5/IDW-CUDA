#pragma once

#include <string>

#include "P2.h"

class Utils {
public:
	static void drawGui(float fps, const std::string& methodName, int mouseValue, double pParam, uint8_t* img, const P2& imgSize, P2 color = P2(255, 0, 0));
	static void drawString(const std::string& str, P2 point, uint8_t* img, const P2 & imgSize, P2 color);
	static void drawChar(char c, const P2& point, uint8_t * img, const P2& imgSize, P2 color);
};
