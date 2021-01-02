#pragma once

#include <iostream>
#include <string>

#define LOG(x) std::cout << x << std::endl


struct P2 {
	int x, y;

	int value =-1;
	
	double norm2d() const;
	
	friend P2 operator+(const P2& left, const P2& right);
	friend P2 operator-(const P2& left, const P2& right);
};