#include "native_bridge.h"
#include <math.h>

BRIDGE_API double fast_dot_product(const double* a, const double* b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

BRIDGE_API void fast_vector_scale(const double* input, double scale, double* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

BRIDGE_API void fast_vector_add(const double* a, const double* b, double* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}

BRIDGE_API double fast_compute_loss(const double* actual, const double* predicted, int size) {
    if (size <= 0) return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = actual[i] - predicted[i];
        sum_sq += diff * diff;
    }
    return sum_sq / (double)size;
}
