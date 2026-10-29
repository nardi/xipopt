"""Prepares the process for the native libraries before the extension module loads.

Linux and macOS find the bundled libraries through the run path that meson-python writes
into the extension module. Windows has no run path, so the directory of the libraries is
added to the search path of DLLs. SPRAL needs OpenMP cancellation, which the OpenMP runtime
reads from the environment when it loads.
"""

import os
import sys
from pathlib import Path

LIBRARY_DIRECTORY = Path(__file__).resolve().parent.parent / ".xipopt.mesonpy.libs"
"""Where meson-python puts the libraries of a wheel, next to the package."""

if sys.platform == "win32":
    if LIBRARY_DIRECTORY.is_dir():
        os.add_dll_directory(str(LIBRARY_DIRECTORY))

os.environ.setdefault("OMP_CANCELLATION", "TRUE")
