#pragma once

struct P3 {
	int x, y, z;

	double norm2d() const;
	
	friend P3 operator+(const P3& left, const P3& right);
	friend P3 operator-(const P3& left, const P3& right);
};