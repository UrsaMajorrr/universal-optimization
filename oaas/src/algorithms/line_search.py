"""
These are numerical optimization methods that follow this criteria:

x_k1 = x_k + alpha*p_k

with alpha being a step length

This includes first order and second order line search
"""
import jax
import jax.numpy as jnp
from jax import grad, hessian
import logging
from typing import List, Optional, Callable

from .utils.step_length import StepLength


class LineSearch:

    def __init__(self, 
                 f: Callable, 
                 x: List[float], 
                 tol: Optional[float] = 1e-6,
                 max_iter: Optional[int] = 1000):
        self.f = f
        self.x = jnp.array(x, dtype=jnp.float64)
        self.tol = tol
        self.max_iter = max_iter

    def gradient_descent(self, lr: float):
        function_evals = 0
        for i in range(self.max_iter):
            grad_f = grad(self.f)(self.x)
            function_evals += 1

            if jnp.linalg.norm(grad_f, ord=jnp.inf) < self.tol:
                break

            self.x = self.x - lr * grad_f
            print(f"Current step: {self.x.tolist()}")

        print(f"Function evals: {function_evals}")
        print(f"Final iteration: {i}")
        print(f"Optimized Solution: {self.x.tolist()}")

        return self.x

    def backtrack_search(self, 
                         alpha: Optional[float] = 10, 
                         rho: Optional[float] = 0.5, 
                         c: Optional[float] = 1e-4):
        function_evals = 0
        for i in range(self.max_iter):
            grad_f = grad(self.f)(self.x)
            p_k = -grad_f / jnp.linalg.norm(grad_f, ord=2)
            function_evals += 1

            if jnp.linalg.norm(grad_f, ord=jnp.inf) < self.tol:
                break

            while self.f(self.x + alpha * p_k) > self.f(self.x) + c * alpha * jnp.dot(grad_f, p_k):
                function_evals += 1
                alpha *= rho

            self.x = self.x + alpha * p_k
            logging.debug(f"Current step: {self.x.tolist()}")
            print(f"Current step: {self.x.tolist()}")

        print(f"Function evals: {function_evals}")
        print(f"Final iteration: {i}")
        print(f"Optimized Solution: {self.x.tolist()}")

        logging.info(f"Function evals: {function_evals}")
        logging.info(f"Final iteration: {i}")
        logging.info(f"Optimized Solution: {self.x.tolist()}")

        return self.x

    def line_search(self):
        
        function_evals = 0
        alpha_obj = StepLength(self.f, self.x)

        for i in range(self.max_iter):
            grad_f = grad(self.f)(self.x)
            p_k = -grad_f
            function_evals += 1

            if jnp.linalg.norm(grad_f, ord=jnp.inf) < self.tol:
                break
            alpha = alpha_obj.step_length(p_k)
            self.x = self.x + alpha * p_k

            logging.debug(f"Current step: {self.x.tolist()}")
            print(f"Current step: {self.x.tolist()}, Current alpha: {alpha}, Iteration: {i}")

        print(f"Function evals: {function_evals}")
        print(f"Final iteration: {i}")
        print(f"Optimized Solution: {self.x.tolist()}")

        logging.info(f"Function evals: {function_evals}")
        logging.info(f"Final iteration: {i}")
        logging.info(f"Optimized Solution: {self.x.tolist()}")

        return self.x

    def newton_line_search(self):
        function_evals = 0
        alpha_obj = StepLength(self.f, self.x)

        for i in range(self.max_iter):
            grad_f = grad(self.f)(self.x)
            hess_f = hessian(self.f)(self.x) + 1e-6 * jnp.eye(self.x.size)
            f_x = self.f(self.x)
            p_k = -jnp.linalg.solve(hess_f, jnp.eye(self.x.shape[0])) @ grad_f
            function_evals += 3

            print(f"Iter {i}: f_x={f_x}, grad_f={grad_f}, ||grad_f||={jnp.linalg.norm(grad_f, ord=jnp.inf)}")
            print(f"p_k: {p_k}")

            if jnp.linalg.norm(grad_f, ord=jnp.inf) < self.tol:
                function_evals += 1
                break
            alpha = alpha_obj.step_length(p_k)
            self.x = self.x + alpha * p_k

            #logging.debug(f"Current step: {self.x.tolist()}")
            print(f"Current step: {self.x.tolist()}, Current alpha: {alpha}")
        print(f"Function evals: {function_evals}")
        print(f"Final iteration: {i}")
        print(f"Optimized Solution: {self.x.tolist()}")

        logging.info(f"Function evals: {function_evals}")
        logging.info(f"Final iteration: {i}")
        logging.info(f"Optimized Solution: {self.x.tolist()}")

        return self.x
