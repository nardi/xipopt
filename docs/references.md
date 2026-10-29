# References

- Wächter, A. and Biegler, L. T. (2006). On the implementation of an interior-point
  filter line-search algorithm for large-scale nonlinear programming. *Mathematical
  Programming* 106(1), 25-57. [DOI](https://doi.org/10.1007/s10107-004-0559-y)
- Hock, W. and Schittkowski, K. (1981). *Test Examples for Nonlinear Programming Codes*.
  Lecture Notes in Economics and Mathematical Systems 187. Springer.
- Gebremedhin, A. H., Manne, F. and Pothen, A. (2005). What color is your Jacobian?
  Graph coloring for computing derivatives. *SIAM Review* 47(4), 629-705.
  [DOI](https://doi.org/10.1137/S0036144504444711)

Software:

- [Ipopt](https://github.com/coin-or/Ipopt), with the [options
  reference](https://coin-or.github.io/Ipopt/OPTIONS.html).
- [asdex](https://pypi.org/project/asdex/), automatic sparse differentiation in JAX.
- [cyxla](https://nardi.github.io/cyxla/), which runs JAX functions from native code.
- [JAX-CIP](https://nardi.github.io/jax-cip/), the same idea for the SCIP solver.
- [meson-jll](https://nardi.github.io/meson-jll/), which turns Julia binary packages
  into Meson wraps.
