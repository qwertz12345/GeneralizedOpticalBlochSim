# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:40:41 2019

@author: Experimentator
"""

from __future__ import division
import numpy as np

from scipy.constants import pi, speed_of_light
from sympy.physics.wigner import wigner_6j

#%%

### Some Constants ###

gamma_D2 = 2 * pi * 6.067e6

omega_D2 = 2 * pi * 384.2304844685e12
omega_D2_f2 = 2 * pi * 384.227921462e12
omega_D2_f1 = 2 * pi * 384.234756145e12

det_fe_0 = -2 * pi * 302.0738e6
det_fe_1 = -2 * pi * 229.8518e6
det_fe_2 = -2 * pi * 72.9112e6
det_fe_3 = 2 * pi * 193.7407e6

lambda_D2 = 2 * pi * speed_of_light / omega_D2
#%% Real Rubidium, D2 line


def alpha_ff(f, fe, j=1.0 / 2.0, je=3.0 / 2.0, i=3.0 / 2.0):
    """
    alpha_f^f' as defined in Gian-Luca's master thesis for large detuning.
    j, je, and i are the default params for rubidium D2 line

    for a spin-1/2 system it would be:
        j = 1/2
        je = 3/2
        i = 0
    """
    return float(
        3.0 / 2.0 * 1.0 * (2.0 * je + 1.0) * wigner_6j(je, fe, i, f, j, 1.0) ** 2
    )


def a0_f_infd(f):
    """
    Calculates the a_f^(0) - value as defined in Gian-Luca's master thesis for infinite detuning for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.

    Returns
    -------
    float
        a_f^(0) number as defined in Gian-luca's Master thesis

    """
    return (
        1.0
        / 3.0
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff(f, f - 1) * (2 * f - 1)
            + alpha_ff(f, f) * (2 * f + 1)
            + alpha_ff(f, f + 1) * (2 * f + 3)
        )
    )


def a1_f_infd(f):
    """
    Calculates the a_f^(1) - value as defined in Gian-Luca's master thesis for infinite detuning for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.

    Returns
    -------
    float
        a_f^(1) number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            -alpha_ff(f, f - 1) * (2 * f - 1) / f
            - alpha_ff(f, f) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff(f, f + 1) * (2 * f + 3) / (f + 1)
        )
    )


def a2_f_infd(f):
    """
    Calculates the a_f^(2) - value as defined in Gian-Luca's master thesis for infinite detuning for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.

    Returns
    -------
    float
        a_f^(2) number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff(f, f - 1) / f
            - alpha_ff(f, f) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff(f, f + 1) / (f + 1)
        )
    )


def a0_infd(f, i, j, je):
    """
    Calculates the a_f^(0) - value as defined in Gian-Luca's master thesis for infinite detuning.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate: Rubidium: 1 or 2, spin-1/2 particle: 1/2
    i : float, integer or half an integer
        i quantum number: Rubidium: 3/2, spin-1/2 particle: 0
    j : float, integer or half an integer
        j quantum number of groundstate: Rubidium: 1/2, spin-1/2 particle: 1/2
    je : float, integer or half an integer
        j quantum number of excited state: D2-line: 3/2, spin-1/2 particle: 1/2.

    Returns
    -------
    float
        a_f^(0) number as defined in Gian-luca's Master thesis

    """
    return (
        1.0
        / 3.0
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff(f, f - 1, j, je, i) * (2 * f - 1)
            + alpha_ff(f, f, j, je, i) * (2 * f + 1)
            + alpha_ff(f, f + 1, j, je, i) * (2 * f + 3)
        )
    )


def a1_infd(f, i, j, je):
    """
    Calculates the a_f^(1) - value as defined in Gian-Luca's master thesis for infinite detuning.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    i : float, integer or half an integer
        i quantum number: Rubidium: 3/2, spin-1/2 particle: 0
    j : float, integer or half an integer
        j quantum number of groundstate: D2-line: 1/2, spin-1/2 particle: 1/2
    je : float, integer or half an integer
        j quantum number of excited state: D2-line: 3/2, spin-1/2 particle: 1/2.

    Returns
    -------
    float
        a_f^(1) number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            -alpha_ff(f, f - 1, j, je, i) * (2 * f - 1) / f
            - alpha_ff(f, f, j, je, i) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff(f, f + 1, j, je, i) * (2 * f + 3) / (f + 1)
        )
    )


def a2_infd(f, i, j, je):
    """
    Calculates the a_f^(2) - value as defined in Gian-Luca's master thesis for infinite detuning.

    I don't understand why this is a finite number for the spin-1/2 particle. There is no Tensor-effect for
    a spin-1/2 particle. But on the other hand, this term does not exist at all for a spin-1/2 particle.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    i : float, integer or half an integer
        i quantum number: Rubidium: 3/2, spin-1/2 particle: 0
    j : float, integer or half an integer
        j quantum number of groundstate: D2-line: 1/2, spin-1/2 particle: 1/2
    je : float, integer or half an integer
        j quantum number of excited state: D2-line: 3/2, spin-1/2 particle: 1/2.

    Returns
    -------
    float
        a_f^(2) number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff(f, f - 1, j, je, i) / f
            - alpha_ff(f, f, j, je, i) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff(f, f + 1, j, je, i) / (f + 1)
        )
    )


def alpha_ff_delta(f, fe, det):
    """
    Calculates the alpha_ff - value as defined in Gian-Luca's master thesis for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    fe : float, integer or half an integer
        f-quantum number of excited state.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.

    Returns
    -------
    float
        alpha_ff number as defined in Gian-luca's Master thesis

    """

    if fe == 0:
        det_ff = det - det_fe_0
    elif fe == 1:
        det_ff = det - det_fe_1
    elif fe == 2:
        det_ff = det - det_fe_2
    elif fe == 3:
        det_ff = det - det_fe_3
    else:
        det_ff = 0

    pre = det_ff * det / (gamma_D2**2 / 4 + det_ff**2)

    return (
        3.0
        / 2.0
        * pre
        * (2.0 * (3.0 / 2.0) + 1.0)
        * float(wigner_6j(3.0 / 2.0, fe, 3.0 / 2.0, f, 1.0 / 2.0, 1.0) ** 2)
    )


def a0_f(f, det):
    """
    Calculates the a_f^0 - value as defined in Gian-Luca's master thesis for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.

    Returns
    -------
    float
        a_f^0 number as defined in Gian-luca's Master thesis

    """
    return (
        1.0
        / 3.0
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff_delta(f, f - 1, det) * (2 * f - 1)
            + alpha_ff_delta(f, f, det) * (2 * f + 1)
            + alpha_ff_delta(f, f + 1, det) * (2 * f + 3)
        )
    )


def a1_f(f, det):
    """
    Calculates the a_f^1 - value as defined in Gian-Luca's master thesis for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundstate f
    to the middle of the excited state manifold as defined in the Alkali D Line data by Steck.

    Returns
    -------
    float
        a_f^1 number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            -alpha_ff_delta(f, f - 1, det) * (2 * f - 1) / f
            - alpha_ff_delta(f, f, det) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff_delta(f, f + 1, det) * (2 * f + 3) / (f + 1)
        )
    )


def a2_f(f, det):
    """
    Calculates the a_f^2 - value as defined in Gian-Luca's master thesis for the Rubidium D2-line.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.

    Returns
    -------
    float
        a_f^2 number as defined in Gian-luca's Master thesis

    """
    return (
        2
        * (-1.0) ** (2.0 * f)
        * (
            alpha_ff_delta(f, f - 1, det) / f
            - alpha_ff_delta(f, f, det) * (2 * f + 1) / (f * (f + 1))
            + alpha_ff_delta(f, f + 1, det) / (f + 1)
        )
    )


def alpha_0(f, det, w0):
    """
    Calculates the scalar polarisability constant (as defined in Gian-Luca's
    master thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float
        alpha_0

    """
    A = pi * w0**2
    return lambda_D2**2 / (2 * pi * A) * gamma_D2 / det * a0_f(f, det)


def alpha_1(f, det, w0):
    """
    Calculates the vector polarisability constant (as defined in Gian-Luca's
    master thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float: alpha_1
    """
    A = pi * w0**2
    return lambda_D2**2 / (8 * pi * A) * gamma_D2 / det * a1_f(f, det)


def alpha_2(f, det, w0):
    """
    Calculates the tensor polarisability constant (as defined in Gian-Luca's
    master thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float
        alpha_2

    """
    A = pi * w0**2
    return lambda_D2**2 / (8 * pi * A) * gamma_D2 / det * a2_f(f, det)


def alpha_0_infd(f, det, w0):
    """
    Calculates the scalar polarisability constant in the limit of large detuning
    (as defined in Gian-Luca's master thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float
        alpha_0

    """
    A = pi * w0**2
    return lambda_D2**2 / (2 * pi * A) * gamma_D2 / det


def alpha_1_infd(f, det, w0):
    """
    Calculates the vector polarisability constant in the limit of large detuning
    (as defined in Gian-Luca's master thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float
        alpha_1

    """
    A = pi * w0**2
    return lambda_D2**2 / (8 * pi * A) * gamma_D2 / det * (-1) ** f


def alpha_2_infd(f, det, w0):
    """
    Calculates the tensor polarisability constant in the limit of large detuning
    (assymptotic expression in Thomas thesis) for the Rubidium D2-line.
    Here, the area is defined as pi*w0^2 - as for a homogenous coupling or the
    coupling to the average intensity of a Gaussian beam.

    Parameters
    ----------
    f : float, integer or half an integer
        f-quantum number of groundstate.
    det : detuning defined as the detuning from one specific groundsate f to the middle of the excited state manifold as
        defined in the Alkali D Line Adata by Steck.
    w0: waist of the probe beam, in meter

    Returns
    -------
    float
        alpha_2

    """
    A = pi * w0**2
    # det_f0 = det - det_fe_0
    # det_f1 = det - det_fe_1
    # det_f2 = det - det_fe_2
    # det_f3 = det - det_fe_3
    if f == 2:
        # delta_fine = (-det_f1 + 5*det_f2 - 4*det_f3)/20
        delta_fine = (-det_fe_1 + 5 * det_fe_2 - 4 * det_fe_3) / 20
    elif f == 1:
        # delta_fine = (-4*det_f0 + 5*det_f1 - det_f2)/4
        delta_fine = -(-4 * det_fe_0 + 5 * det_fe_1 - det_fe_2) / 4
    else:
        raise ("The groundstate has to be f = 1 or 2.")
    return lambda_D2**2 / (8 * pi * A) * gamma_D2 * delta_fine / det**2


#%%
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    w0 = 50e-6
    f = 2
    x_min = 0.1
    x_max = 100
    x = np.geomspace(x_min, x_max, 1000)
    det_sign = -1

    a0 = np.abs(alpha_0(f, det_sign * 2 * pi * x * 1e9, w0))
    a1 = np.abs(alpha_1(f, det_sign * 2 * pi * x * 1e9, w0))
    a2 = np.abs(alpha_2(f, det_sign * 2 * pi * x * 1e9, w0))

    a0_inf = np.abs(alpha_0_infd(f, det_sign * 2 * pi * x * 1e9, w0))
    a1_inf = np.abs(alpha_1_infd(f, det_sign * 2 * pi * x * 1e9, w0))
    a2_inf = np.abs(alpha_2_infd(f, det_sign * 2 * pi * x * 1e9, w0))

    plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(
        hspace=None, wspace=None, top=0.9, bottom=0.2, left=0.2, right=0.9
    )
    plt.loglog(x, a0, "-", color=[0, 0.8, 0.5], label=r"$|\alpha_0|$")
    plt.loglog(x, a1, "-.", color=[0, 0, 0.8], label=r"$|\alpha_1|$")
    plt.loglog(x, a2, "--", color=[0.8, 0, 0], label=r"$|\alpha_2|$")

    plt.loglog(
        x,
        a0_inf,
        "-",
        color=[0, 0.8, 0.5, 0.5],
        label=r"$|\alpha_0|$ for large $\Delta$",
    )
    plt.loglog(
        x,
        a1_inf,
        "-.",
        color=[0, 0, 0.8, 0.5],
        label=r"$|\alpha_1|$ for large $\Delta$",
    )
    plt.loglog(
        x,
        a2_inf,
        "--",
        color=[0.8, 0, 0, 0.5],
        label=r"$|\alpha_2|$ for large $\Delta$",
    )

    plt.ylabel("Coefficient")
    plt.xlabel(str(det_sign) + r" $ |\Delta|/(2\pi)$ (GHz)")
    plt.title("f = %i" % f)
    plt.grid()
    plt.xlim([x_min, x_max])
    plt.legend(loc=0)
