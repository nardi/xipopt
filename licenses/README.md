# Licenses of bundled libraries

The wheels of xipopt contain shared libraries from the Julia binary builders. The license
of xipopt itself is in `LICENSE.md` of the repository. The libraries keep their own
licenses, and this directory holds a copy of each.

| Library | License | File |
| --- | --- | --- |
| Ipopt | Eclipse Public License 2.0 | `Ipopt-LICENSE` |
| MUMPS | CeCILL-C, with parts under the BSD 3-clause license | `MUMPS_seq-LICENSE` |
| SPRAL | BSD 3-clause | `SPRAL-LICENCE` |
| METIS | Apache License 2.0 | `METIS-LICENSE-2.0.txt` |
| OpenBLAS | BSD 3-clause | `OpenBLAS32-LICENSE` |
| libblastrampoline | MIT | `libblastrampoline-LICENSE.md` |
| ASL | MIT-like | `ASL-LICENSE.txt` |
| hwloc | BSD 3-clause | `Hwloc-COPYING` |
| libxml2 | MIT | `XML2-Copyright` |
| libpciaccess | MIT | `Xorg_libpciaccess-COPYING` |
| zlib | zlib license | `Zlib-README` |
| libiconv | LGPL 2.1 | `Libiconv-COPYING` |
| GCC runtime libraries | GPL 3.0 with the GCC runtime library exception | `CompilerSupportLibraries-GPL-3.0+` |

The Ipopt license requires that its source stays available. The source is at
https://github.com/coin-or/Ipopt, and the binaries come from the
[Ipopt_jll](https://github.com/JuliaBinaryWrappers/Ipopt_jll.jl) package. The other
libraries have a `_jll` package of their own in the same organization.
