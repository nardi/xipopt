"""Makes the shared libraries that a wheel bundles loadable on Windows.

Linux and macOS find them through the run path that meson-python writes into the extension
module. Windows has no run path, so the directory has to be added to the search path of
DLLs before the extension module loads.
"""

import os
import sys
from pathlib import Path

LIBRARY_DIRECTORY = Path(__file__).resolve().parent.parent / ".xipopt.mesonpy.libs"
"""Where meson-python puts the libraries of a wheel, next to the package."""

if sys.platform == "win32":
    if LIBRARY_DIRECTORY.is_dir():
        os.add_dll_directory(str(LIBRARY_DIRECTORY))
