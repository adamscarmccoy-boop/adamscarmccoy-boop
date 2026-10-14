#ifndef NATIVE_BRIDGE_H
#define NATIVE_BRIDGE_H

#ifdef _WIN32
#define BRIDGE_API __declspec(dllexport)
#else
#define BRIDGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

BRIDGE_API double fast_dot_product(const double* a, const double* b, int size);
BRIDGE_API void fast_vector_scale(const double* input, double scale, double* output, int size);
BRIDGE_API void fast_vector_add(const double* a, const double* b, double* output, int size);
BRIDGE_API double fast_compute_loss(const double* actual, const double* predicted, int size);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_BRIDGE_H
