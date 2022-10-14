from genobs.Rubidium87_operators import *


def faraday_rot_angle(rho: Qobj, detuning) -> float:
    """faraday angle d2 for the mw sensing setup, far detuned

    Parameters
    ----------
        rho (Qobj): density matrix
        detuning (float): Detuning/2pi with respect to the transition with the smallest energy (F=2->F'=1).

    Returns
    -------
        Float: angle
    """
    wavelength_probe_laser = 780e-9
    density_atoms = 2.33e12 / (1e-2) ** 3
    length_cell = 2e-3
    detunings_probe = [(detuning - 6.834682e9) * 2 * pi, detuning * 2 * pi]
    return (
        sum(
            [
                mF
                * (-1) ** F
                * rho.matrix_element(ket_Fg_D1(F, mF).dag(), ket_Fg_D1(F, mF))
                / detunings_probe[F - 1]
                for F in (1, 2)
                for mF in range(-F, F + 1)
            ]
        ).real
        * density_atoms
        * length_cell
        * wavelength_probe_laser**2
        * GAMMA_RAD_D2
        * 0.5
    )


def Fz_mean(rho: Qobj) -> tuple:
    """Calculates the means of the z-component of the collective spin operator F
    in the ground state for F=1 and F=2

    Parameters
    ----------
    rho : Qobj
        density matrix
    """
    populations_ground = rho.diag()[:8]
    F1z = [mf * populations_ground[mf + 1] for mf in (-1, 0, 1)]
    F2z = [mf * populations_ground[mf + 2 + 3] for mf in (-2, -1, 0, 1, 2)]
    return sum(F1z), sum(F2z)


def F1_F2_rot_angle(detuning, F1zmean, F2zmean, c=(3 * GAMMA_RAD_D2) ** 2):
    def rb87_atom_density(temperature_kelvin):
        return (
            np.power(10, 2.881 + 4.312 - 4040 / temperature_kelvin)
            * 133.3224
            / (1.380649e-23 * temperature_kelvin)
            * 1e-6
            * 0.75
        )  # /cm^3

    wavelength = 780.241209686e-9 * 100  # cm
    density = rb87_atom_density(363.15)  # per cm^3
    cell_length = 0.2  # cm
    common_factor = (
        0.5
        * 3
        * wavelength**2
        * density
        * cell_length
        * GAMMA_RAD_D2
        / (8 * np.pi)
        * 1000
    )
    D2_CENTER = 230.4844685 - 0.165234
    D2_F1_TRANSITION = 4.271676631815181 + D2_CENTER
    GROUND_STATE_HFS = 6.834682610904290
    D2_F2_TRANSITION = D2_F1_TRANSITION - GROUND_STATE_HFS
    D2_DELTA_FP3 = 193.7407e-3
    D2_DELTA_FP2 = -72.9112e-3
    D2_DELTA_FP1 = -229.8518e-3
    D2_DELTA_FP0 = -302.0738e-3
    detuning = detuning - D2_F1_TRANSITION + D2_CENTER
    return common_factor * (
        F1zmean
        * (
            -(detuning - D2_DELTA_FP0)
            * (2 * np.pi)
            / (3 * (((detuning - D2_DELTA_FP0) * (2 * np.pi)) ** 2 + c))
            - 5
            * (detuning - D2_DELTA_FP1)
            * (2 * np.pi)
            / (12 * (((detuning - D2_DELTA_FP1) * (2 * np.pi)) ** 2 + c))
            + 5
            * (detuning - D2_DELTA_FP2)
            * (2 * np.pi)
            / (12 * (((detuning - D2_DELTA_FP2) * (2 * np.pi)) ** 2 + c))
        )
        + F2zmean
        * (
            -(detuning + GROUND_STATE_HFS - D2_DELTA_FP1)
            * (2 * np.pi)
            / (20 * (((detuning + GROUND_STATE_HFS - D2_DELTA_FP1) * (2 * np.pi)) ** 2 + c))
            - (detuning + GROUND_STATE_HFS - D2_DELTA_FP2)
            * (2 * np.pi)
            / (12 * (((detuning + GROUND_STATE_HFS - D2_DELTA_FP2) * (2 * np.pi)) ** 2 + c))
            + 7
            * (detuning + GROUND_STATE_HFS - D2_DELTA_FP3)
            * (2 * np.pi)
            / (15 * (((detuning + GROUND_STATE_HFS - D2_DELTA_FP3) * (2 * np.pi)) ** 2 + c))
        )
    )
