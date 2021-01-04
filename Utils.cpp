#include "Utils.h"

#include "FontDefinition.h"
#include <fmt/core.h>

void Utils::drawGui(float fps, const std::string& methodName, int mouseValue, double pParam, uint8_t* img, const P2& imgSize, const uint8_t color) {

	P2 point(imgSize.x - 30 * 8, imgSize.y - 13 - 5);
	drawString("Vladislav Trnka 2020", point, img, imgSize, color);

	point -= P2{ 0, 25 };
	drawString(fmt::format("{:<12}: {}", "Method", methodName), point, img, imgSize, color);
	
	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "FPS", fps), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "P param", pParam), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Wheel value", mouseValue), point, img, imgSize, color);
	
}

void Utils::drawString(const std::string& str, P2 point, uint8_t* img, const P2& imgSize, const uint8_t color) {

	for (const auto c : str) {
		drawChar(c, point, img, imgSize, color);
		point.x += 9;

		if (point.x + 8 > imgSize.x) break;
		if (point.y + 13 > imgSize.y) break;
	}
}

void Utils::drawChar(const char c, const P2& point, uint8_t* img, const P2& imgSize, const uint8_t color) {
	auto* letter = letters[c - ' '];

	for (int i = 0; i < 13; ++i) {

		char mask = 0b00000001;
		for (int bit = 7; bit >= 0; bit--) {

			if (letter[i] & mask) {
				img[(point.y + i) * imgSize.x + point.x + bit] = color;
			}

			mask <<= 1;
		}
	}
}


void Utils::drawGui(float fps, const std::string& methodName, int mouseValue, double pParam, uint32_t* img, const P2& imgSize, const uint32_t rgbaColor) {

	P2 point(imgSize.x - 30 * 8, imgSize.y - 13 - 5);
	drawString("Vladislav Trnka 2020", point, img, imgSize, rgbaColor);

	point -= P2{ 0, 25 };
	drawString(fmt::format("{:<12}: {}", "Method", methodName), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "FPS", fps), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "P param", pParam), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Wheel value", mouseValue), point, img, imgSize, rgbaColor);

}

void Utils::drawString(const std::string& str, P2 point, uint32_t* img, const P2& imgSize, const uint32_t rgbaColor) {

	for (const auto c : str) {
		drawChar(c, point, img, imgSize, rgbaColor);
		point.x += 9;

		if (point.x + 8 > imgSize.x) break;
		if (point.y + 13 > imgSize.y) break;
	}
}

void Utils::drawChar(const char c, const P2& point, uint32_t* img, const P2& imgSize, const uint32_t rgbaColor) {
	auto* letter = letters[c - ' '];

	for (int i = 0; i < 13; ++i) {

		char mask = 0b00000001;
		for (int bit = 7; bit >= 0; bit--) {

			if (letter[i] & mask) {
				img[(point.y + i) * imgSize.x + point.x + bit] = rgbaColor;
			}

			mask <<= 1;
		}
	}
}