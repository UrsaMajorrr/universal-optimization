import jax
import jax.numpy as jnp
from jax import grad, hessian
import logging
from typing import List, Optional, Callable

from .utils.step_length import StepLength

class ConjugateGradient:
    """
    A class of algorithms for conjugate gradient methods
    """

    def __init__(self, 
                 f: Callable, 
                 x: List[float], 
                 tol: Optional[float]=1e-6, 
                 max_iter: Optional[int]=1000):
        self.f = f
        self.x = jnp.array(x, dtype=jnp.float64)
        self.tol = tol
        self.max_iter = max_iter

    def linear_conjuage_gradient(self, 
                                 A: List[List[float]], 
                                 b: List[float]):
        """
        Used for solving linear systems
        """
        b = jnp.array(b, dtype=jnp.float64)
        r = jnp.matmul(A, self.x) - b
        p = -1*r
        alpha = 0
        beta = 0

        for _ in range(self.max_iter):
            alpha = (jnp.transpose(r) @ r)/(jnp.transpose(p) @ A @ p + 1e-6)
            self.x = self.x + alpha*p
            r_k1 = r - alpha*A @ p

            if r < self.tol:
                break

            beta = (jnp.transpose(r_k1) @ r_k1)/(jnp.transpose(r) @ r + 1e-6)
            p = r_k1 + beta*p
            r = r_k1

        return self.x
    
    def CGFR(self):
        g = grad(self.f)(self.x)
        p = -g
        beta = 0
        function_evals = 0
        alpha_obj = StepLength(self.f)

        for i in range(self.max_iter):
            alpha = alpha_obj.step_length(p, self.x)
            self.x = self.x + alpha*p

            g_1 = grad(self.f)(self.x)
            function_evals += 1

            if jnp.linalg.norm(g_1, ord=jnp.inf) < self.tol:
                break

            beta = jnp.dot(jnp.transpose(g_1), g_1) / (jnp.dot(jnp.transpose(g), g) + 1e-6)
            p = -g_1 + beta*p
            g = g_1
            logging.debug(f"Current step: {self.x.tolist()}")
            print(f"Current step: {self.x.tolist()}, Current alpha: {alpha}, Iteration: {i}")

        print(f"Function evals: {function_evals}")
        print(f"Final iteration: {i}")
        print(f"Optimized Solution: {self.x.tolist()}")

        logging.info(f"Function evals: {function_evals}")
        logging.info(f"Final iteration: {i}")
        logging.info(f"Optimized Solution: {self.x.tolist()}")


        return self.x
