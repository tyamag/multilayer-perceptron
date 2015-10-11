#include "MathUtil.hpp"

#include <cmath>

namespace ccilab {

double Sigmoid(const double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace ccilab
