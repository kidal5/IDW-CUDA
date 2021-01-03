#include "Utils.h"

#include "FontDefinition.h"
#include <fmt/core.h>

void Utils::drawGui(float fps, const std::string& methodName, int mouseValue, uint8_t* img, const P2& imgSize, const P2 color) {

	P2 point(imgSize.x - 25 * 8, imgSize.y - 13 - 5);
	drawString("Vladislav Trnka 2020", point, img, imgSize, color);

	point -= P2{ 0, 25 };
	drawString(fmt::format("Method: {}", methodName), point, img, imgSize, color);
	
	point -= P2{ 0, 15 };
	drawString(fmt::format("FPS: {:.5f}", fps), point, img, imgSize, color);

	point -= P2{ 0, 15 };
	drawString(fmt::format("Wheel value: {}", mouseValue), point, img, imgSize, color);
	
}

void Utils::drawString(const std::string& str, P2 point, uint8_t* img, const P2& imgSize, const P2 color) {

	for (const auto c : str) {
		drawChar(c, point, img, color);
		point.x += 9;

		if (point.x + 8 > imgSize.x) break;
		if (point.y + 13 > imgSize.y) break;
	}
}

void Utils::drawChar(const char c, const P2& point, uint8_t* img, const P2 color) {
	auto* letter = letters[c - ' '];

	const int width = 768;

	for (int i = 0; i < 13; ++i) {

		char mask = 0b00000001;
		for (int bit = 7; bit >= 0; bit--) {

			if (letter[i] & mask) {
				img[3 * ((point.y + i) * width + point.x + bit) + 0] = color.x;
				img[3 * ((point.y + i) * width + point.x + bit) + 1] = color.y;
				img[3 * ((point.y + i) * width + point.x + bit) + 2] = color.value;
			}

			mask <<= 1;
		}
	}
}
