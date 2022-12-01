from typing import List
from scipy.constants import pi, c, epsilon_0, hbar
from qutip import Qobj, basis, tensor, spin_Jx, spin_Jy, spin_Jz, qeye
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j, clebsch_gordan


# We set hbar = 1
A_S = 3.417341305452145e09 * 2 * pi  # Hz
A_P12 = 407.25e6 * 2 * pi
A_P32 = 84.7185e6 * 2 * pi
B_P32 = 12.4965e6 * 2 * pi
MU_BOHR = 2 * pi * 1.399624604e6  # Hz / G
I = 3 / 2  # Rb87
JG = 1 / 2
JD1 = 1 / 2
JD2 = 3 / 2

OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL = 4.4876 * 1e-3 / (1e-2) ** 2
SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING = 1.66933 * (1e-3) / (1e-2) ** 2


def K_factor(F, J):
    return F * (F + 1) - I * (I + 1) - J * (J + 1)


def Eg(F):
    Jg = 1 / 2
    return 1 / 2 * A_S * K_factor(F, Jg)


def Delta_E_P12(F):
    Je = 1 / 2
    return 1 / 2 * A_P12 * K_factor(F, Je)


def Delta_E_P32(F):
    Je = 3 / 2
    K = K_factor(F, Je)
    return 1 / 2 * A_P32 * K + B_P32 * (
        3 / 2 * K * (K + 1) - 2 * I * (I + 1) * Je * (Je + 1)
    ) / (4 * I * (2 * I - 1) * Je * (2 * Je - 1))


def basic_ket_Fg(F, mF) -> Qobj:
    assert F in (1, 2)
    assert abs(mF) <= F
    if F == 1:
        return basis(8, mF + F)
    elif F == 2:
        return basis(8, mF + F + 3)
    else:
        raise ValueError


def ket_Fg_D1(F, mF) -> Qobj:
    return Qobj(
        np.concatenate(
            (basic_ket_Fg(F, mF).full().flatten(), np.zeros(8, dtype=np.cdouble))
        )
    )


def ket_Fg_D2(F, mF) -> Qobj:
    return Qobj(
        np.concatenate(
            (basic_ket_Fg(F, mF).full().flatten(), np.zeros(16, dtype=np.cdouble))
        )
    )


def basic_ket_Fe_D1(F, mF) -> Qobj:
    assert F in (1, 2)
    assert abs(mF) <= F
    if F == 1:
        return basis(8, mF + F)
    elif F == 2:
        return basis(8, mF + F + 3)
    else:
        raise ValueError


def basic_ket_Fe_D2(F, mF) -> Qobj:
    assert F in (0, 1, 2, 3)
    assert abs(mF) <= F
    if F == 0:
        return basis(16, 0)
    elif F == 1:
        return basis(16, mF + F + 1)
    elif F == 2:
        return basis(16, mF + F + 4)
    elif F == 3:
        return basis(16, mF + F + 9)
    else:
        raise ValueError


def ket_Fe_D2(F, mF) -> Qobj:
    return Qobj(
        np.concatenate(
            (np.zeros(8, dtype=np.cdouble), basic_ket_Fe_D2(F, mF).full().flatten())
        )
    )


def ket_Fe_D1(F, mF) -> Qobj:
    return Qobj(
        np.concatenate(
            (np.zeros(8, dtype=np.cdouble), basic_ket_Fe_D1(F, mF).full().flatten())
        )
    )


def H_atom(det_Light, line: str) -> Qobj:
    sum_Fg = sum(
        [Eg(f) * basic_ket_Fg(f, m).proj() for f in (1, 2) for m in range(-f, f + 1)]
    )
    if line == "D1":
        h = np.zeros(shape=(16, 16), dtype=np.cdouble)
        sum_Fe_D1 = sum(
            [
                (Delta_E_P12(f) - det_Light) * basic_ket_Fe_D1(f, m).proj()
                for f in (1, 2)
                for m in range(-f, f + 1)
            ]
        )
        h[:8, :8] = sum_Fg
        h[8:, 8:] = sum_Fe_D1
        return Qobj(h)
    elif line == "D2":
        h = np.zeros(shape=(24, 24), dtype=np.cdouble)
        sum_Fe_D2 = sum(
            [
                (Delta_E_P32(f) - det_Light) * basic_ket_Fe_D2(f, m).proj()
                for f in (0, 1, 2, 3)
                for m in range(-f, f + 1)
            ]
        )
        h[:8, :8] = sum_Fg
        h[8:, 8:] = sum_Fe_D2
        return Qobj(h)
    else:
        raise ValueError


def weight_lowering_op(Fg, mg, Fe, me, Je, q):
    return (
        (-1) ** (Fe + JG + 1 + I)
        * ((2 * Fe + 1) * (2 * JG + 1)) ** (1 / 2)
        * float(clebsch_gordan(Fe, 1, Fg, me, q, mg))
        * float(wigner_6j(Je, JG, 1, Fg, Fe, I))
    )


def sigma_q(q: int, line: str) -> Qobj:
    """ "weighted lowering operator", m_F = m'_F + q"""
    assert q in (-1, 0, 1)
    assert line in ("D1", "D2")
    Jg = 1 / 2
    if line == "D1":
        Je = 1 / 2
        opers = []
        for Fg in (1, 2):
            for mg in range(-Fg, Fg + 1):
                for Fe in (1, 2):
                    for me in range(-Fe, Fe + 1):
                        weight = (
                            (-1) ** (Fe + Jg + 1 + I)
                            * ((2 * Fe + 1) * (2 * Jg + 1)) ** (1 / 2)
                            * float(clebsch_gordan(Fe, 1, Fg, me, q, mg))
                            * float(wigner_6j(Je, Jg, 1, Fg, Fe, I))
                        )
                        if weight != 0.0:
                            lowering_operator = (
                                basic_ket_Fg(Fg, mg) * basic_ket_Fe_D1(Fe, me).dag()
                            )
                            opers.append(weight * lowering_operator)
        tmp = np.zeros(shape=(16, 16), dtype=np.cdouble)
        tmp[:8, 8:] = (sum(opers)).full()
        return Qobj(tmp)
    elif line == "D2":
        Je = 3 / 2
        opers = []
        for Fg in (1, 2):
            for mg in range(-Fg, Fg + 1):
                for Fe in (0, 1, 2, 3):
                    for me in range(-Fe, Fe + 1):
                        weight = (
                            (-1) ** (Fe + Jg + 1 + I)
                            * ((2 * Fe + 1) * (2 * Jg + 1)) ** (1 / 2)
                            * float(clebsch_gordan(Fe, 1, Fg, me, q, mg))
                            * float(wigner_6j(Je, Jg, 1, Fg, Fe, I))
                        )
                        if weight != 0.0:
                            lowering_operator = (
                                basic_ket_Fg(Fg, mg) * basic_ket_Fe_D2(Fe, me).dag()
                            )
                            opers.append(weight * lowering_operator)
        tmp = np.zeros(shape=(24, 24), dtype=np.cdouble)
        tmp[:8, 8:] = (sum(opers)).full()
        return Qobj(tmp)
    else:
        raise ValueError


def E_0_plus(intensity):
    # we assume pure polarization and either sigma plus or minus or pi pol
    return (2 * intensity / (c * epsilon_0)) ** (1 / 2) / 2


def calculate_beam_intensity(power, waist):
    """
    Watt, m
    """
    return power / (pi * waist**2)


def rabi_D1_vector_component(E_field_component):
    dipole = 2.5377e-29  # dipole transition matrix element
    return 2 * dipole * E_field_component / hbar


def rabi_D2_vector_component(E_field_component):
    dipole = 3.58424e-29  # dipole transition matrix element
    return 2 * dipole * E_field_component / hbar


def H_atom_field_D1(q, E_field_component) -> Qobj:
    tmp = (
        1
        / 2
        * np.conjugate(rabi_D1_vector_component(E_field_component))
        * sigma_q(q, "D1")
    )
    return tmp + tmp.dag()


def H_atom_field_D2(q, E_field_component) -> Qobj:
    tmp = (
        1
        / 2
        * np.conjugate(rabi_D2_vector_component(E_field_component))
        * sigma_q(q, "D2")
    )
    return tmp + tmp.dag()


def H_hfs_ground() -> Qobj:
    """in basis: I, J=S"""
    return A_S * (
        tensor(-spin_Jz(1 / 2), -spin_Jz(3 / 2))  # I * J (S_1/2, where L=0)
        + tensor(
            -spin_Jy(1 / 2), -spin_Jy(3 / 2)
        )  # different order of states? (+2,+1...)
        + tensor(spin_Jx(1 / 2), spin_Jx(3 / 2))
    )


def H_hfs_excited_D1() -> Qobj:
    """in basis I, J"""
    return A_P12 * (
        # different order of states? (+2,+1...)
        tensor(-spin_Jz(1 / 2), -spin_Jz(3 / 2))
        + tensor(-spin_Jy(1 / 2), -spin_Jy(3 / 2))
        + tensor(spin_Jx(1 / 2), spin_Jx(3 / 2))
    )


def H_hfs_excited_D2() -> Qobj:
    """in basis J, I"""
    I_dot_J = (
        #  J                        I
        tensor(-spin_Jz(3 / 2), -spin_Jz(3 / 2))
        + tensor(-spin_Jy(3 / 2), -spin_Jy(3 / 2))
        + tensor(spin_Jx(3 / 2), spin_Jx(3 / 2))
    )
    return A_P32 * I_dot_J + B_P32 * (
        (3 * I_dot_J * I_dot_J + 3 / 2 * I_dot_J - 3 / 2 * (5 / 2) * 3 / 2 * 5 / 2)
        / (2 * 3 / 2 * (3 - 1) * 3 / 2 * (3 - 1))
    )


# def H_hfs_excited_ISL_D2() -> Qobj:
#     """in basis I,S,L"""
#     I_dot_J = (
#         tensor(
#             spin_Jz(3 / 2), spin_Jz(1 / 2), qeye(3)
#         )  # different order of states? (+2,+1...)
#         + tensor(spin_Jy(3 / 2), spin_Jy(1 / 2), qeye(3))
#         + tensor(spin_Jx(3 / 2), spin_Jx(1 / 2), qeye(3))
#         + tensor(spin_Jz(3 / 2), qeye(2), spin_Jz(1))
#         + tensor(spin_Jy(3 / 2), qeye(2), spin_Jy(1))
#         + tensor(spin_Jx(3 / 2), qeye(2), spin_Jx(1))
#     )
#     return A_P32 * I_dot_J + B_P32 * (
#         (3 * I_dot_J * I_dot_J + 3 / 2 * I_dot_J - 3 / 2 * (5 / 2) * 3 / 2 * 5 / 2)
#         / (2 * 3 / 2 * (3 - 1) * 3 / 2 * (3 - 1))
#     )


def H_B_ground_uncoupled(bx=0, by=0, bz=0) -> Qobj:
    """
    In Gauss
    in basis of I, J=S (L=0)
    returns Qobj with dimensions [4, 2]
    """
    return (
        2.0023193043622
        * (  #     S            I
            tensor(spin_Jx(1 / 2), qeye(4))
            * bx  # different order of states? (+2,+1...)
            + tensor(-spin_Jy(1 / 2), qeye(4))
            * by  # different order of states? (+2,+1...)
            + tensor(-spin_Jz(1 / 2), qeye(4))
            * bz  # different order of states? (+2,+1...)
        )
        - 0.000995
        * (
            tensor(qeye(2), spin_Jx(3 / 2))
            * bx  # different order of states? (+2,+1...)
            + tensor(qeye(2), -spin_Jy(3 / 2))
            * by  # different order of states? (+2,+1...)
            + tensor(qeye(2), -spin_Jz(3 / 2))
            * bz  # different order of states? (+2,+1...)
        )
    ) * MU_BOHR


def H_B_excited_uncoupled(bx=0, by=0, bz=0) -> Qobj:  # L=1, in G
    """
    L=1, in G
    in uncoupled basis of L, S, I with dimension 4*2*3 = 24.
    D1 and D2 excited state manifolds
    """
    return (
        2.0023193043622
        * (  # L           S           I
            tensor(qeye(3), spin_Jx(1 / 2), qeye(4)) * bx
            + tensor(qeye(3), -spin_Jy(1 / 2), qeye(4)) * by
            + tensor(qeye(3), -spin_Jz(1 / 2), qeye(4)) * bz
        )
        - 0.000995
        * (
            tensor(qeye(3), qeye(2), spin_Jx(3 / 2)) * bx
            + tensor(qeye(3), qeye(2), -spin_Jy(3 / 2)) * by
            + tensor(qeye(3), qeye(2), -spin_Jz(3 / 2)) * bz
        )
        + 0.99999369
        * (
            tensor(spin_Jx(1), qeye(2), qeye(4)) * bx
            + tensor(-spin_Jy(1), qeye(2), qeye(4)) * by
            + tensor(-spin_Jz(1), qeye(2), qeye(4)) * bz
        )
    ) * MU_BOHR


def mJ_mI_uncoupled_basis_excited():
    """
    List of
    |J' mJ, mI> = clebsch * |L=1 mL, mS, mI>. (dim = 3*2*4=24)
    First eight kets: J'=1/2,
    the next 16: J'=3/2.
    """
    out = []
    for J in (1 / 2, 3 / 2):
        for mj in np.arange(-J, J + 1):
            for mi in (-3 / 2, -1 / 2, 1 / 2, 3 / 2):
                vs = [
                    tensor(
                        sum(
                            [
                                float(clebsch_gordan(1, 1 / 2, J, ml, ms, mj))
                                * tensor(basis(3, ml + 1), basis(2, int(ms + 1 / 2)))
                                for ml in (-1, 0, 1)
                                for ms in (-1 / 2, 1 / 2)
                            ]
                        ),
                        basis(4, int(mi + 3 / 2)),
                    )
                ]
                out += vs
    return out


def Fp_kets_in_JI_basis(J) -> List:
    """ """
    coupled = []
    for F in np.arange(3 / 2 - J, 3 / 2 + J + 1):
        for m in range(-int(F), int(F) + 1):
            coup_vec = sum(
                [
                    float(clebsch_gordan(J, 3 / 2, F, j, i, m))
                    * tensor(
                        basis(int(2 * J + 1), int(j + J)), basis(4, int(i + 3 / 2))
                    )
                    for j in np.arange(-J, J + 1)
                    for i in np.arange(-3 / 2, 3 / 2 + 1)
                ]
            )
            coupled.append(coup_vec)
    return coupled


def F_kets_in_JI_basis():
    """
    Returns all 8 ground-state F kets (L=0).
    Dimensions:
    """
    fm_states = []
    S = 1 / 2  # = J
    for f in np.arange(3 / 2 - S, 3 / 2 + S + 1):
        for m in np.arange(-f, f + 1):
            fm_state = sum(
                [
                    float(clebsch_gordan(S, 3 / 2, f, s, i, m))
                    * tensor(basis(2, int(s + 1 / 2)), basis(4, int(i + 3 / 2)))
                    for s in (-1 / 2, 1 / 2)
                    for i in np.arange(-3 / 2, 3 / 2 + 1)
                ]
            )
            fm_states.append(fm_state)
    return fm_states


def Bxy_from_mw_rabi_sigma_plus_minus(rabi_sigma_plus, rabi_sigma_minus):
    r"""
    MW Field Polarization
    Calculates Bx, By from Rabi frequencies at Mw sigma + and sigma - transitions:
    |1, 1 > with |2, 2 > and |1, -1 > with |2, -2 >
    """
    mu = 7628535.3802797925  # = abs(H_B("D1", bx=1)[0, 3])
    return (
        (rabi_sigma_plus - rabi_sigma_minus) / (4 * mu),  # ?
        (rabi_sigma_plus + rabi_sigma_minus) / (4 * mu),
    )


def Bz_from_rabi_pi_clock(rabi_pi):
    return rabi_pi / 8808673.910654247


GAMMA_RAD_D1 = 5.7500e6 * 2 * pi
GAMMA_RAD_D2 = 6.0666e6 * 2 * pi


def natural_decay_ops_D2():
    return [(2 * GAMMA_RAD_D2) ** (1 / 2) * sigma_q(q, "D2") for q in (-1, 0, 1)]


def natural_decay_ops_D1():
    return [GAMMA_RAD_D1 ** (1 / 2) * sigma_q(q, "D1") for q in [-1, 0, 1]]


# def sigma_q_jbasis_D2(q):


# def LS_basis_in_J_basis_excited_states() -> List:
#     """
#     Returns List of Quantum objects with  dims = [[6], [1]], shape = (6, 1),
#     type = ket
#     """
#     kets = []
#     for j in (1 / 2, 3 / 2):
#         for ml in (-1, 0, 1):
#             for ms in (-1 / 2, 1 / 2):
#                 v = sum(
#                     [
#                         float(clebsch_gordan(1, 1 / 2, j, ml, ms, mj))
#                         * basis(int(2*j+1), int(mj + j))
#                         for mj in np.arange(-j, j + 1)
#                     ]
#                 )
#                 if v.norm()>1e-9:
#                     kets.append(v)
#     return kets


# def H_B_excited_IJ(bx=0, by=0, bz=0) -> Qobj:
#     return Qobj(H_B_excited_uncoupled(bx=bx, by=by, bz=bz).full()).transform(
#         [Qobj(elem.full()) for elem in uncoupled_basis_in_IJ_basis_excited_states()]
#     )


# def J_basis_in_LS_basis(J=3 / 2) -> List:
#     return [
#         sum(
#             [
#                 float(clebsch_gordan(1 / 2, 1, J, ms, ml, mj))
#                 * tensor(basis(3, ml + 1), basis(2, int(ms + 1 / 2)))
#                 for ml in (-1, 0, 1)
#                 for ms in (-1 / 2, 1 / 2)
#             ]
#         )
#         for mj in np.arange(-J, J + 1)
#     ]


# def H_B_excited_D2_IJ_basis(bx=0, by=0, bz=0):
#     """
#     Quantum object: dims = [[16], [16]]
#     """
#     hb = H_B_excited_uncoupled(bx=bx, by=by, bz=bz)
#     bv = J_basis_in_SL_basis(J=3 / 2)
#     ibv = [tensor(bv[k], basis(4, i)) for i in range(4) for k in range(4)]
#     hbt = np.zeros((16, 16), dtype=np.cdouble)
#     for n in range(16):
#         for m in range(16):
#             hbt[n, m] = (ibv[n].dag() * hb * ibv[m])[0, 0]  # n, m order correct?
#     return Qobj(hbt)


# def F_basis_in_IJ_D2():
#     fp_ij = Fp_kets_in_I_J_basis(J=3 / 2)
#     f_ij = F_kets_in_I_J_basis()
#     new_vectors = [np.zeros(24, dtype=np.cdouble) for _ in range(24)]
#     for i in range(8):
#         new_vectors[i][:8] = f_ij[i].full().flatten()
#     for i in range(8, 24):
#         new_vectors[i][8:] = fp_ij[i - 8].full().flatten()
#     new_kets = [Qobj(v) for v in new_vectors]
#     return new_kets


# def H_B_D2_F_basis(bx=0, by=0, bz=0):
#     be = H_B_excited_D2_IJ_basis(bx=bx, by=by, bz=bz)
#     bet = be.transform(Fp_kets_in_I_J_basis(J=3 / 2)).tidyup(atol=1e-3)
#     bg = H_B_ground_uncoupled(bx=bx, by=by, bz=bz)
#     bgt = bg.transform(F_kets_in_I_J_basis()).tidyup(atol=1e-3)
#     b = np.zeros((24, 24), dtype=np.cdouble)
#     b[:8, :8] = bgt.full()
#     b[8:, 8:] = bet.full()
#     return Qobj(b)


def H_Bz_excited_D2_JI(bz):
    return (
        (
            -0.000995 * tensor(qeye(4), -spin_Jz(3 / 2))
            + 1.3362 * tensor(-spin_Jz(3 / 2), qeye(4))
        )
        * bz
        * MU_BOHR
    )


def H_Bz_excited_D1_IJ(bz):
    return (
        (
            -0.000995 * tensor(-spin_Jz(3 / 2), qeye(2))
            + 0.666 * tensor(qeye(4), -spin_Jz(1 / 2))
        )
        * bz
        * MU_BOHR
    )
