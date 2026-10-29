/* The operating system calls of the extension module, so the rest of the code is portable. */
#ifndef XIPOPT_PLATFORM_H
#define XIPOPT_PLATFORM_H

#ifdef _WIN32
/*
 * Windows has no `dladdr`, and `windows.h` cannot be included here: the include directories
 * of the bundled libraries come first, and the one of hwloc has a header of that name. The
 * Python side finds the path of a library by an address with `ctypes` instead.
 */
static const char* xipopt_library_path(const void* address) {
  (void)address;
  return NULL;
}
#else
#include <dlfcn.h>

/* Returns the path of the library that holds an address, or NULL if there is none. */
static const char* xipopt_library_path(const void* address) {
  Dl_info info;
  if (!dladdr(address, &info)) {
    return NULL;
  }
  return info.dli_fname;
}
#endif

#endif
