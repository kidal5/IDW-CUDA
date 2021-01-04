#include "Utils.h"

#include "Constants.h"
#include "DataManager.h"
#include "CpuIdwBase.h"
#include "FontDefinition.h"
#include <fmt/core.h>

#include "GL/wglext.h"
typedef BOOL(APIENTRY* PFNWGLSWAPINTERVALFARPROC)(int);
PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;


void Utils::drawGui(DataManager& data, CpuIdwBase& base) {

	
	if (base.getMethodName().find("OpenGL") != std::string::npos) {
		printGui(base.getFps(), data.getFullFPS(), base.getMethodName(), data.getMouseValue(), data.getAnchorPoints().size(), data.getPParam(), data.getCurrentPalette().name);
		return;
	}
	
	if (data.getCurrentPalette().isEightBit) {
		drawGui(base.getFps(), data.getFullFPS(), base.getMethodName(), data.getMouseValue(), data.getAnchorPoints().size(), data.getPParam(), data.getCurrentPalette().name, base.getBitmapGreyscaleCpu(), imgSize, 150);
	} else {
		drawGui(base.getFps(), data.getFullFPS(), base.getMethodName(), data.getMouseValue(), data.getAnchorPoints().size(), data.getPParam(), data.getCurrentPalette().name, base.getBitmapColorCpu(), imgSize, 0xFF000000);
	}
}

void Utils::printGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName) {

	fmt::print("{:<12}: {}\n", "Method", methodName);
	fmt::print("{:<12}: {:.2f}\n", "FPS", fps);
	fmt::print("{:<12}: {:.2f}\n", "Real FPS", fullFps);
	fmt::print("{:<12}: {:.2f}\n", "P param", pParam);
	fmt::print("{:<12}: {}\n", "Points count", pointsCount);
	fmt::print("{:<12}: {}\n", "Wheel value", mouseValue);
	fmt::print("{:<12}: {}\n\n\n", "Palette", paletteName);

}

void Utils::drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint8_t* img, const P2& imgSize, const uint8_t color) {

	P2 point(imgSize.x - 30 * 8, imgSize.y - 13 - 5);
	drawString("Vladislav Trnka 2020", point, img, imgSize, color);

	point -= P2{ 0, 25 };
	drawString(fmt::format("{:<12}: {}", "Method", methodName), point, img, imgSize, color);
	
	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "FPS", fps), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "Real FPS", fullFps), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "P param", pParam), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Points count", pointsCount), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Wheel value", mouseValue), point, img, imgSize, color);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Palette", paletteName), point, img, imgSize, color);
	
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


void Utils::drawGui(float fps, float fullFps, const std::string& methodName, int mouseValue, int pointsCount, double pParam, const std::string& paletteName, uint32_t* img, const P2& imgSize, const uint32_t rgbaColor) {

	P2 point(imgSize.x - 30 * 8, imgSize.y - 13 - 5);
	drawString("Vladislav Trnka 2020", point, img, imgSize, rgbaColor);

	point -= P2{ 0, 25 };
	drawString(fmt::format("{:<12}: {}", "Method", methodName), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "FPS", fps), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "Real FPS", fullFps), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {:.2f}", "P param", pParam), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Points count", pointsCount), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Wheel value", mouseValue), point, img, imgSize, rgbaColor);

	point -= P2{ 0, 16 };
	drawString(fmt::format("{:<12}: {}", "Palette", paletteName), point, img, imgSize, rgbaColor);
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

int Utils::getBiggerPowerOfTwo(const int num) {
	int power = 1;
	while (power < num)
		power *= 2;

	return power;
}

void Utils::setVSync(const int interval) {

	wglSwapIntervalEXT = reinterpret_cast<PFNWGLSWAPINTERVALFARPROC>(wglGetProcAddress("wglSwapIntervalEXT"));
	if (wglSwapIntervalEXT && wglSwapIntervalEXT(interval))
		fmt::print("Vsync disabled\n");
}

