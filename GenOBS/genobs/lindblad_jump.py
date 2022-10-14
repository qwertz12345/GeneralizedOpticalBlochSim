from scipy.constants import pi

GAMMA_RAD_D1 = 5.7500e6 * 2 * pi
GAMMA_RAD_D2 = 6.0666e6 * 2 * pi
QUENCHING_RATE = 8.4e7


def natural_decay_ops_D2():
    return [(2 * GAMMA_RAD_D2) ** (1 / 2) * sigma_q(q, "D2") for q in (-1, 0, 1)]


def natural_decay_ops_D1():
    return [GAMMA_RAD_D1 ** (1 / 2) * sigma_q(q, "D1") for q in [-1, 0, 1]]


def quenching_ops(line: str, gamma=QUENCHING_RATE):
    if line == "D1":
        return [
            (gamma) ** (1 / 2)
            * get_ket_Fg_D1(fg, mg)
            * get_ket_Fe_D1(fe, me).dag()
            for fg in (1, 2)
            for mg in range(-fg, fg + 1)
            for fe in (1, 2)
            for me in range(-fe, fe + 1)
        ]
    elif line == "D2":
        return [
            (gamma) ** (1 / 2)
            * get_ket_Fg_D2(fg, mg)
            * get_ket_Fe_D2(fe, me).dag()
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
            (gamma) ** (1 / 2)
            * get_ket_Fg_D1(f, mb)
            * get_ket_Fg_D1(f, ma).dag()
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
            (gamma) ** (1 / 2)
            * get_ket_Fg_D2(f, mb)
            * get_ket_Fg_D2(f, ma).dag()
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
            (gamma) ** (1 / 2)
            * get_ket_Fg_D1(1, mf)
            * get_ket_Fg_D1(2, ms).dag()
            for mf in (-1, 0, 1)
            #     ]
            # )
            for ms in range(-2, 3)
        ]
    elif line == "D2":
        return [
            # sum(
            #     [
            (gamma) ** (1 / 2)
            * get_ket_Fg_D2(1, mf)
            * get_ket_Fg_D2(2, ms).dag()
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
        (gamma) ** (1 / 2) * ket_Fg(2, ms) * ket_Fg(1, mf).dag()
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


def wall_coll(line, gamma=1e3):
    dims_op = 16 if line == "D1" else 24
    return [
        (gamma) ** (1 / 2)
        * basis(dims_op, end)
        * basis(dims_op, begin).dag()
        for end in range(8)
        for begin in range(8)
        # if begin != end
    ]


def dephasing_excited_states(line: str, gamma=1e7):  # value for rate???
    if line == "D1":
        return [
            (gamma) ** (1 / 2) * get_ket_Fe_D1(f, mf).proj()
            for f in (1, 2)
            for mf in range(-f, f + 1)
        ]
    elif line == "D2":
        return [
            (gamma) ** (1 / 2) * get_ket_Fe_D2(f, mf).proj()
            for f in range(4)
            for mf in range(-f, f + 1)
        ]


def dephasing_ground_states(line: str, gamma=1e3):
    if line == "D1":
        return [
            (gamma) ** (1 / 2) * get_ket_Fg_D1(f, mf).proj()
            for f in (1, 2)
            for mf in range(-f, f + 1)
        ]
    elif line == "D2":
        return [
            (gamma) ** (1 / 2) * get_ket_Fg_D2(f, mf).proj()
            for f in (1, 2)
            for mf in range(-f, f + 1)
        ]