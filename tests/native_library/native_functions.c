/* Native functions for the tests of xipopt. */

#include <stddef.h>
#include <stdint.h>

#include "cyxla/native.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* results[0] = arguments[0] squared. */
EXPORT const char* square(void* user_data, int64_t argument_count,
                          const CyxlaNativeArray* arguments, int64_t result_count,
                          CyxlaNativeArray* results, int64_t attribute_count,
                          const CyxlaNativeAttribute* attributes) {
  int64_t count = 1;
  for (int64_t axis = 0; axis < arguments[0].rank; ++axis) count *= arguments[0].dimensions[axis];
  const double* source = (const double*)arguments[0].data;
  double* target = (double*)results[0].data;
  for (int64_t index = 0; index < count; ++index) target[index] = source[index] * source[index];
  return NULL;
}
