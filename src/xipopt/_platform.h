/* The operating system calls of the extension modules, so the rest of the code is portable. */
#ifndef XIPOPT_PLATFORM_H
#define XIPOPT_PLATFORM_H

#ifdef _WIN32
#include <windows.h>

/* Returns the path of the library that holds an address, or NULL if there is none. */
static const char* xipopt_library_path(const void* address) {
  static char path[MAX_PATH];
  HMODULE module = NULL;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)address, &module)) {
    return NULL;
  }
  if (GetModuleFileNameA(module, path, MAX_PATH) == 0) {
    return NULL;
  }
  return path;
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
