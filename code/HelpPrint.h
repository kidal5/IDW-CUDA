#pragma once

/// Helper class for printing user interface help
class HelpPrint {
public:

	/// Print program user interface int console
	static void print();

	/// Print program user interface int console when key is h
	static bool handleKeys(const unsigned char key, const int x, const int y);
	
};
