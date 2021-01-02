#include "P3.h"

#include <cmath>

P3 operator+ (const P3& left, const P3& right) {
    return P3{ left.x + right.x,left.y + right.y, left.z + right.z};
}

P3 operator- (const P3& left, const P3& right) {
    return P3{ left.x - right.x,left.y - right.y, left.z - right.z };
}

double P3::norm2d() const {
	return std::sqrt(x * x + y * y);
}
