#include "P2.h"

#include <cmath>


P2 operator+ (const P2& left, const P2& right) {
    return P2{ left.x + right.x,left.y + right.y };
}

P2 operator- (const P2& left, const P2& right) {
    return P2{ left.x - right.x,left.y - right.y };
}

double P2::norm2d() const {
    return std::sqrt(x * x + y * y);
}