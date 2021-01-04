#include "HelpPrint.h"

#include <fmt/core.h>
#include <fmt/color.h>

const std::string keys[26]{
	"ESC",
	"f",
	"h",
	"v",
	"",
	"ARROW UP",
	"ARROW DOWN",
	"ARROW LEFT",
	"ARROW RIGHT",
	"TAB",
	"",
	"c",
	"r",
	"",
	"0",
	"+",
	"",
	"1",
	"2",
	"3",
	"4",
	"5",
	"6",
	"7",
	"8",
	"9",
};

const std::string keysText[26]{
	"Close the program",
	"Toggle fullscreen",
	"Print this message",
	"Toggle VSync - vertical synchronization - enables more fps than 60",
	"",
	"Increment p param",
	"Decrement p param",
	"Change palette to the right",
	"Change palette to the left",
	"Move to next IDW computation method",
	"",
	"Clear all anchor points",
	"Generate random anchor points",
	"",
	"Save current anchor points as user data",
	"Read user data",
	"",
	"Read data - 50  points / resolution  256x256",
	"Read data - 100 points / resolution  256x256",
	"Read data - 150 points / resolution  256x256",
	"Read data - 200 points / resolution  768x768",
	"Read data - 300 points / resolution  768x768",
	"Read data - 400 points / resolution  768x768",
	"Read data - 100 points / resolution 1920x1080",
	"Read data - 200 points / resolution 1920x1080",
	"Read data - 500 points / resolution 1920x1080",
};

void HelpPrint::print() {
	handleKeys('h', 0, 0);
}

bool HelpPrint::handleKeys(const unsigned char key, const int x, const int y) {

	if (key != 'h') return false;
	
	//resize window ???
	fmt::print(fg(fmt::color::green), "Welcome to IDW visualizer by Vladislav Trnka\n");
	fmt::print("Created in year 2020 as university term job\n");
	fmt::print("\n");

	fmt::print(fg(fmt::color::green), "Mouse functions\n");
	fmt::print("{:<17}: {}\n", "Left click", "Add anchor point");
	fmt::print("{:<17}: {}\n", "Right click", "Delete anchor point");
	fmt::print("{:<17}: {}\n", "Wheel up/down", "Change value of added point");
	fmt::print("\n");

	fmt::print(fg(fmt::color::green), "Keys functions\n");

	for (int i = 0; i < 26; ++i) {
		if (keys[i].empty()) {
			fmt::print("\n");
		} else {
			fmt::print("{:<17}: {}\n", keys[i], keysText[i]);
		}
	}
	fmt::print("\n");

	fmt::print(fg(fmt::color::green), "Have a nice day!\n");

	return true;
	
}
