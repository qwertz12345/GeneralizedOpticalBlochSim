# %%
from typing import List

import numpy as np
from genobs.experiment_parameters import *
from genobs.Rubidium87_operators import *
from genobs.visualizations import *
from qutip import *
from copy import copy
from lmfit.model import Model

# %%
def hamil(
    mw_det,
    b_longitudinal=0.1,
    B_mw_vector=[0, 0, 0],
    laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL / 500,
):
    ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()
    ham_laser_atom[:3, :] = 0  # F=1 -> F' neglected
    ham_laser_atom[:, :3] = 0
    hb_ac = H_B("D1", bx=B_mw_vector[0], by=B_mw_vector[1], bz=B_mw_vector[2])
    hb_ac = hb_ac.full()
    for k in range(16):
        hb_ac[k, k] = 0.0  # rotating...
    hb_ac[8:, 8:] = 0.0
    h0 = H_atom(0, "D1") + H_B("D1", bz=b_longitudinal)
    # h0 = h0.full()
    h0 = np.diag(h0.diag())
    ham_tot = Qobj(h0) + Qobj(hb_ac) + Qobj(ham_laser_atom)
    temp = ham_tot.full()
    laser_freq = temp[9, 9] - temp[5, 5]
    for k in range(8, 16):
        temp[k, k] -= laser_freq
    diff_f2_f1 = temp[5, 5] - temp[1, 1]
    for k in range(3):
        temp[k, k] += diff_f2_f1  # rotating frame mw: level of F=1 shifted to F=2
    for k in range(3):
        temp[k, k] += mw_det
    for i in range(15):  # RWA rf freqs
        temp[i, i + 1] = 0.0
        temp[i + 1, i] = 0.0
    en_offset = temp[1, 1]
    for i in range(16):
        temp -= en_offset  # set |F=1, mF=0 > as zero energy
    return Qobj(temp)


def P_wire(b_mw, distance=0.03):
    from scipy import constants

    return (b_mw * 1e-4 / constants.mu_0 * distance * 2 * pi) ** 2 * 50


# %%


def hamiltonians_all_transitions(bvector, laser_intens=0.01 * 10, b_longitudinal=0.1):
    """Hamiltonians for 7 hyperfine transitions (2 double transitions).
    The steady states for MW-off case is also calculated.

    Parameters
    ----------
    bvector : tuple
        mw b  vector
    laser_intens : Laer intensity in W/m², optional
        _description_, by default 0.01*10
    b_longitudinal : Bz static, optional
        _description_, by default 0.1

    Returns
    -------
    tuple
        list of hamiltonias and the laser steady state
    """
    ham_mw_off = hamil(
        0,
        b_longitudinal=b_longitudinal,
        laser_intens=laser_intens,
        B_mw_vector=[0, 0, 0],
    )
    laser_ss = steadystate(ham_mw_off, c_op_list=decays)
    # plot_bar_excited_pop_D1(laser_ss)
    # plot_bar_ground_pop(laser_ss)
    hyperfine_transition_freqs = [
        ham_mw_off[3, 3] - ham_mw_off[0, 0],  # MW sigma minus
        ham_mw_off[4, 4] - ham_mw_off[0, 0],  # pi 1
        (ham_mw_off[4, 4] - ham_mw_off[1, 1]) / 2
        + (ham_mw_off[5, 5] - ham_mw_off[0, 0]) / 2,  # double
        ham_mw_off[5, 5] - ham_mw_off[1, 1],  # pi clock
        (ham_mw_off[5, 5] - ham_mw_off[2, 2]) / 2
        + (ham_mw_off[6, 6] - ham_mw_off[1, 1]) / 2,  # double
        ham_mw_off[6, 6] - ham_mw_off[2, 2],
        ham_mw_off[7, 7] - ham_mw_off[2, 2],
    ]
    hams = [
        hamil(
            mw_detuning, b_longitudinal, laser_intens=laser_intens, B_mw_vector=bvector
        )
        for mw_detuning in hyperfine_transition_freqs
    ]
    # mw_ss = steadystate(h, c_op_list=decays)
    # plot_bar_excited_pop_D1(mw_ss-laser_ss)
    # plot_bar_ground_pop(mw_ss-laser_ss)
    return hams, laser_ss


#%%
decays = (
    natural_decay_ops_D1()
    + quenching_ops("D1")
    + wall_coll("D1", gamma=2e3)  # gamma chosen to roughly approx experiment
    + dephasing_excited_states("D1", gamma=1.6e8)
    # + dephasing_ground_states_D1()
)
bx, by = Bxy_from_mw_rabi_sigma_plus_minus(80.871301e3, 67.557761e3)  # from experiment
bz = Bz_from_rabi_pi_clock(74.693683e3)
all_hamils_laser_ss = []
laser_intensities = [0.01 * 1, 0.01 * 10, 5 * 0.01 * 10, 0.01 * 100, 0.01 * 1000]
for intens in laser_intensities:
    hs, laser_ss = hamiltonians_all_transitions((bx, by, bz), laser_intens=intens)
    all_hamils_laser_ss += [(h, laser_ss, decays) for h in hs]


#%%
import ipyparallel as ipp
from sim import run_simulation

nr_cores = 2
cluster = ipp.Cluster(n=nr_cores)
cluster.start_cluster_sync()
rc = cluster.connect_client_sync()
rc.wait_for_engines(nr_cores)
rc.ids
dview = rc[:]
parallel_result = dview.map_sync(run_simulation, all_hamils_laser_ss)

for k, intens in enumerate(laser_intensities):
    for i in range(7):
        qsave(parallel_result[k * 7 + i], f"transition_{i+1}_{k}")
import pandas as pd

pd.DataFrame({"laser_intensities": laser_intensities}).to_csv("laser_intensities.csv")
pd.DataFrame({"B mw": (bx, by, bz), "b longitudinal": 0.1}).to_csv(
    "other_parameters.csv"
)
