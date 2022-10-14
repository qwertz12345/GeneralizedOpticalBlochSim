# %%
from genobs.lib import *


def H_hfs_excited_D2():
    """in uncoupled basis !"""
    I_times_J = (
        tensor(spin_Jz(3 / 2), spin_Jz(3 / 2))
        + tensor(spin_Jy(3 / 2), spin_Jy(3 / 2))
        + tensor(spin_Jx(3 / 2), spin_Jx(3 / 2))
    )
    return A_P32 * I_times_J + B_P32 * (
        (
            3 * I_times_J * I_times_J
            + 3 / 2 * I_times_J
            - 3 / 2 * (5 / 2) * 3 / 2 * 5 / 2
        )
        / (2 * 3 / 2 * (3 - 1) * 3 / 2 * (3 - 1))
    )


# %%
eigv, eigs = H_hfs_ground().eigenstates()
eigve, eigse = H_hfs_excited_D2().eigenstates()
# %%


def H_B_excited(bx=0, by=0, bz=0):  # L=1, in G
    """in uncoupled basis !"""
    return (
        2.0023193043622
        * (  #          I           S            L
            tensor(qeye(4), spin_Jx(1 / 2), qeye(3)) * bx
            + tensor(qeye(4), spin_Jy(1 / 2), qeye(3)) * by
            + tensor(qeye(4), spin_Jz(1 / 2), qeye(3)) * bz
        )
        - 0.000995
        * (
            tensor(spin_Jx(3 / 2), qeye(2), qeye(3)) * bx
            + tensor(spin_Jy(3 / 2), qeye(2), qeye(3)) * by
            + tensor(spin_Jz(3 / 2), qeye(2), qeye(3)) * bz
        )
        + 0.99999369
        * (
            tensor(qeye(4), qeye(2), spin_Jx(1)) * bx
            + tensor(qeye(4), qeye(2), spin_Jy(1)) * by
            + tensor(qeye(4), qeye(2), spin_Jz(1)) * bz
        )
    ) * MU_BOHR


# %%
coupled = []
for J in (1 / 2, 3 / 2):
    for j in np.arange(-J, J + 1):
        coup_vec = sum(
            [
                clebsch(1 / 2, 1, J, s, l, j)
                * tensor(basis(2, int(s + 1 / 2)), basis(3, l + 1))
                for l in (-1, 0, 1)
                for s in (-1 / 2, 1 / 2)
            ]
        )
        coupled.append(coup_vec)

# %%
fm_states = []
for J in (1 / 2, 3 / 2):
    for f in np.arange(3 / 2 - J, 3 / 2 + J + 1):
        for m in np.arange(-f, f + 1):
            fm_state = sum(
                [
                    clebsch(3 / 2, J, f, i, j, m)
                    * clebsch(1 / 2, 1, J, s, l, j)
                    * tensor(
                        basis(4, int(i + 3 / 2)),
                        basis(2, int(s + 1 / 2)),
                        basis(3, l + 1),
                    )
                    for l in (-1, 0, 1)
                    for s in (-1 / 2, 1 / 2)
                    for i in np.arange(-3 / 2, 3 / 2 + 1)
                    for j in np.arange(-J, J + 1)
                ]
            )
            fm_states.append(fm_state)
# %%
H_B_excited(bz=1).transform(fm_states) / 2 / pi

# %%
