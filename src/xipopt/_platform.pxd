# The operating system calls of `_platform.h`, which only `_solver` uses.


cdef extern from "_platform.h" nogil:
    const char* xipopt_library_path(const void* address)
