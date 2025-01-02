from src.algorithms.line_search import LineSearch
from src.algorithms.conjugate_gradient import ConjugateGradient
import jax.numpy as jnp
from jax import grad, hessian

x = [6.0, 10.0]
f = lambda x: (x[1] - 5.1 / (4 * jnp.pi**2) * x[0]**2 + 5 * x[0] / jnp.pi - 6)**2 + 10 * (1 - 1 / (8 * jnp.pi)) * jnp.cos(x[0]) + 10

# Uncomment for a simpler function
# x = [3.0]
# f = lambda x: x[0]**2 + 2*x[0]

tol = 1e-6
max_iter = 100

ls = LineSearch(f, x, tol=tol, max_iter=max_iter)
# ls.gradient_descent(lr=0.001)
# ls.backtrack_search()
# ls.line_search()
# ls.newton_line_search()

cg = ConjugateGradient(f, x, tol=tol, max_iter=max_iter)
cg.CGFR()