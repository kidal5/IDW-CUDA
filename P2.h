#pragma once

struct P2 {

	P2(int _x, int _y) : x(_x), y(_y){};
	P2(int _x, int _y, int _z) : x(_x), y(_y), value(_z) {}

	int x, y;

	int value =-1;
	
	double norm2d() const;
	
	P2& operator+=(const P2& rhs);
	P2& operator-=(const P2& rhs);

	friend P2 operator+(const P2& left, const P2& right);
	friend P2 operator-(const P2& left, const P2& right);
};