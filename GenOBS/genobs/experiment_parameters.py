from scipy.constants import pi
from genobs.Rubidium87_operators import *

PROBE_BEAM_WAIST = (
    ((1.552 + 1.613 + 1.593 + 1.606) / 4) / 2 * 1e-3
)  # in m, waist radius
PUMP_BEAM_WAIST = ((2.476 + 2.644 + 2.426 + 2.611) / 4) / 2 * 1e-3  # in m, waist radius


def get_probe_intensity(power):
    """power in W, returns intensity in W/m**2"""
    return power / ((PROBE_BEAM_WAIST) ** 2 * pi)


def get_pump_intensity(power):
    """power in W, returns intensity in W/m**2"""
    return power / ((PUMP_BEAM_WAIST) ** 2 * pi)


QUENCHING_RATE = 8.4e7


def quenching_ops(line: str, gamma=QUENCHING_RATE):
    if line == "D1":
        return [
            (gamma) ** (1 / 2) * ket_Fg_D1(fg, mg) * ket_Fe_D1(fe, me).dag()
            for fg in (1, 2)
            for mg in range(-fg, fg + 1)
            for fe in (1, 2)
            for me in range(-fe, fe + 1)
        ]
    elif line == "D2":
        return [
            (gamma) ** (1 / 2) * ket_Fg_D2(fg, mg) * ket_Fe_D2(fe, me).dag()
            for fg in (1, 2)
            for mg in range(-fg, fg + 1)
            for fe in (0, 1, 2, 3)
            for me in range(-fe, fe + 1)
        ]
    else:
        raise ValueError


def intra_F_ground_decay(line: str, gamma=5e3):
    if line == "D1":
        return [
            # sum(
            #     [
            (gamma) ** (1 / 2) * ket_Fg_D1(f, mb) * ket_Fg_D1(f, ma).dag()
            for f in (1, 2)
            for mb in range(-f, f + 1)
            #     ]
            # )
            for ma in range(-f, f + 1)
        ]
    elif line == "D2":
        return [
            # sum(
            #     [
            (gamma) ** (1 / 2) * ket_Fg_D2(f, mb) * ket_Fg_D2(f, ma).dag()
            for f in (1, 2)
            for mb in range(-f, f + 1)
            #     ]
            # )
            for ma in range(-f, f + 1)
        ]
    else:
        raise ValueError


# def intra_F_excited_decay(line: str, gamma=1e4):


def F2_to_F1_ground_state_decay(line: str, gamma=3e3) -> List:
    if line == "D1":
        return [
            # sum(
            #     [
            (gamma) ** (1 / 2) * ket_Fg_D1(1, mf) * ket_Fg_D1(2, ms).dag()
            for mf in (-1, 0, 1)
            #     ]
            # )
            for ms in range(-2, 3)
        ]
    elif line == "D2":
        return [
            # sum(
            #     [
            (gamma) ** (1 / 2) * ket_Fg_D2(1, mf) * ket_Fg_D2(2, ms).dag()
            for mf in (-1, 0, 1)
            #     ]
            # )
            for ms in range(-2, 3)
        ]
    else:
        raise ValueError


def F1_to_F2_ground_state_decay(line: str, gamma=3e3) -> List:
    tmp = [
        # sum(
        #     [
        (gamma) ** (1 / 2) * basic_ket_Fg(2, ms) * basic_ket_Fg(1, mf).dag()
        for ms in range(-2, 3)
        #     ]
        # )
        for mf in (-1, 0, 1)
    ]
    dims_op = 16 if line == "D1" else 24
    ops = []
    for op in tmp:
        c = np.zeros(shape=(dims_op, dims_op), dtype=np.cdouble)
        c[:8, :8] = op.full()
        ops.append(Qobj(c))
    return ops


def wall_coll(line, gamma=2e3):
    dims_op = 16 if line == "D1" else 24
    return [
        (gamma) ** (1 / 2) * basis(dims_op, end) * basis(dims_op, begin).dag()
        for end in range(8)
        for begin in range(8)
        # if begin != end
    ]


def dephasing_excited_states(line: str, gamma=1.6e8):
    if line == "D1":
        f_list = (1, 2)
        ket_f = ket_Fe_D1
    elif line == "D2":
        f_list = (0, 1, 2, 3)
        ket_f = ket_Fe_D2
    return [
        (gamma / 2) ** (1 / 2) * (ket_f(f, mf).proj() - ket_f(fs, ms).proj())
        for f in f_list
        for mf in range(-f, f + 1)
        for fs in f_list
        for ms in range(-fs, fs + 1)
    ]


def dephasing_ground_states(line: str, gamma=1e3):
    if line == "D1":
        f_list = (1, 2)
        ket_f = ket_Fg_D1
    elif line == "D2":
        f_list = (0, 1, 2, 3)
        ket_f = ket_Fg_D2
    return [
        (gamma / 2) ** (1 / 2) * (ket_f(f, mf).proj() - ket_f(fs, ms).proj())
        for f in f_list
        for mf in range(-f, f + 1)
        for fs in f_list
        for ms in range(-fs, fs + 1)
    ]


def faraday_rot_angle(rho, detuning):
    """Faraday Rotation Angle in the usual approximation for a
    linearly polarized laser (D2).

    Parameters
    ----------
    rho : Qobj
        Densitiy matrix
    detuning_ : float
        Detuning from D2 Center (384.230 484 468 5 THz)

    Returns
    -------
    Float
        Rotation Angle
    """
    ground_state_pops = rho.diag()[:8]
    wavelength_probe_laser = 780e-9
    density_atoms = 2.33e12 / (1e-2) ** 3
    length_cell = 2e-3
    detunings_hfs = [
        detuning - 4.271676631815181e9 * 2 * pi,
        detuning + 2.563005979089109e9 * 2 * pi,
    ]
    return (
        sum(
            [
                mF * (-1) ** F
                # * rho.matrix_element(ket_Fg_D1(F, mF), ket_Fg_D1(F, mF))
                * ground_state_pops[mF + F]
                if F == 1
                else ground_state_pops[mF + F + 3] / detunings_hfs[F - 1]
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
