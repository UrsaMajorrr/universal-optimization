import jax
import jax.numpy as jnp
from typing import List, Callable
from jax import grad

class StepLength():

    def __init__(self, 
                 f: Callable):
        self.f = f

    def step_length(self,
                        p_k: List[float],
                        x: List[float],                     
                        alpha_max: float = 10.0, 
                        c1: float = 1e-4, 
                        c2: float = 0.9):
            alpha_0 = 0.0
            alpha_1 = jax.random.uniform(jax.random.PRNGKey(0), minval=0.0, maxval=alpha_max)
            phi = self.f(x + alpha_1 * p_k)
            phi_prime = jnp.dot(grad(self.f)(x + alpha_1 * p_k), p_k)
            
            f_x = self.f(x)
            grad_f = grad(self.f)(x)

            if (phi > f_x + c1 * alpha_1 * phi_prime) or (phi >= self.f(x + alpha_0 * p_k)):
                alpha = self.zoom(x, alpha_0, alpha_1, p_k, c1, c2)
            elif abs(phi_prime) <= -c2 * jnp.dot(grad_f, p_k):
                alpha = alpha_1
            elif phi_prime >= 0:
                alpha = self.zoom(x, alpha_1, alpha_0, p_k, c1, c2)
            else:
                alpha = alpha_1

            return alpha

    def zoom(self, x, alpha_low, alpha_high, p_k, c1, c2, tol: float = 1e-6):
        while abs(alpha_high - alpha_low) > tol:
            alpha_j = (alpha_low + alpha_high) / 2

            phi_alpha_j = self.f(x + alpha_j * p_k)
            phi_alpha_low = self.f(x + alpha_low * p_k)
            grad_f = grad(self.f)(x)
            phi_prime_alpha_j = jnp.dot(grad(self.f)(x + alpha_j * p_k), p_k)

            if phi_alpha_j > self.f(x) + c1 * alpha_j * jnp.dot(grad_f, p_k) or phi_alpha_j >= phi_alpha_low:
                alpha_high = alpha_j
            else:
                if abs(phi_prime_alpha_j) <= -c2 * jnp.dot(grad_f, p_k):
                    return alpha_j

                if phi_prime_alpha_j * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low

                alpha_low = alpha_j

            print(f"Zooming: alpha_low={alpha_low}, alpha_high={alpha_high}, alpha_j={alpha_j}")

        return (alpha_low + alpha_high) / 2