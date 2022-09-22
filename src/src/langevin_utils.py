from typing import Callable

import jax
import jax.numpy as jnp


# Set up notation to match Appendix B.1 of https://arxiv.org/pdf/2112.07068.pdf
# Note that during training then in the notation of the paper we have v0 = 0,
# Σxx0 = 0, Σvv0 = γM (= self.initial_velocity * self.mass)
# Which is reflected in the calculations here.

def langevin_mean(
    friction: float,
    int_beta: Callable[[float], float],
    t: float,
    data: float,
):
    B = int_beta(t)
    Γ = friction
    y0 = data

    exp_m2 = jnp.exp(-2 * B * Γ**-1)

    μyt = (2 * B * Γ**-1 + 1) * exp_m2 * y0
    μvt = -B * exp_m2 * y0
    return μyt, μvt


def _langevin_std_small_t(B, Γ, Σvv0):
    # Computes Lyyt, Lyvt, Lvvt as an affine function of B
    # (found via Taylor expansion of _langevin_std_large_t in sympy)
    expm1_m4 = -4*B*Γ**-1 + 8*B**2*Γ**-2
    exp_m4 = 1 - 4*B*Γ**-1 + 8*B**2*Γ**-2

    Σyyt = 8 * B * Γ**-1  # + O(B**2)
    Σyvt = 4 * B * Σvv0 * Γ**-2  # + O(B**2)
    Σvvt = Σvv0 + B * (2 * Γ - 8 * Σvv0 * Γ**-1)  # + O(B**2)

    Lyyt = 2 * jnp.sqrt(2 * B*Γ**-1) * (1 + B * (-2 * Γ**-1 + Σvv0 * Γ**-3))  # + O(B**2)
    Lyvt = jnp.sqrt(B) * (Σvv0 * jnp.sqrt(2 * Γ**-3) + B * (jnp.sqrt(2 * Γ**-1) + Σvv0 * jnp.sqrt(2) * (2 * Γ**-2.5 - Σvv0 * Γ**-4.5) - 6 * Σvv0 * jnp.sqrt(2 * Γ**-5)))  # + O(B**2)
    Lvvt2 = Σvv0 + B * (2 * Γ + Σvv0 * (4 * Γ**-1 - 2 * Σvv0 * Γ**-3) - 12 * Σvv0 * Γ**-1)  # + O(B**2)
    Lvvt = jnp.sqrt(Σvv0) * (1 + B * (Γ * Σvv0**-1 - 4 * Γ**-1 - Σvv0 * Γ**-3))
    return Σyyt, Σyvt, Σvvt, Lyyt, Lyvt, Lvvt, Lvvt2



def _langevin_std_large_t(B, Γ, Σvv0):
    expm1_m4 = jnp.expm1(-4 * B * Γ**-1)
    exp_m4 = jnp.exp(-4 * B * Γ**-1)

    Σyyt = -expm1_m4 + (4 * B * Γ**-1 - 8 * B**2 * Γ**-2 + 16 * B**2 * Γ**-4 * Σvv0) * exp_m4
    Σyvt = (4 * B * Γ**-2 * Σvv0 + 4 * B**2 * Γ**-1 - 8 * B**2 * Γ**-3 * Σvv0) * exp_m4
    Σvvt = -0.25 * Γ**2 * expm1_m4 + (B * Γ + Σvv0 * (1 + 4 * B**2 * Γ**-2 - 4 * B * Γ**-1) - 2 * B**2) * exp_m4

    Lyyt = jnp.sqrt(Σyyt)
    Lyvt = Σyvt / Lyyt
    Lvvt2 = (Σyyt * Σvvt - Σyvt**2) / Σyyt
    Lvvt = jnp.sqrt(Lvvt2)
    return Σyyt, Σyvt, Σvvt, Lyyt, Lyvt, Lvvt, Lvvt2


def langevin_std(
    friction: float,
    mass: float,
    initial_velocity: float,
    int_beta: Callable[[float], float],
    t: float,
):
    B = int_beta(t)
    B_clipped = int_beta(jnp.maximum(t, 1e-4))  # "double-where trick"
    Γ = friction
    Σvv0 = initial_velocity * mass
    pred = t < 1e-4
    small = _langevin_std_small_t(B, Γ, Σvv0)
    large = _langevin_std_large_t(B_clipped, Γ, Σvv0)
    keep = lambda a, b: jnp.where(pred, a, b)
    return jax.tree_map(keep, small, large)
