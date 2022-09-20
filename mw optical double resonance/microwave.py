# %%
from lmfit.models import LorentzianModel, ConstantModel
from init import *
decays = (
    natural_decay_ops_D1() +
    quenching_ops("D1") +
    wall_coll("D1", gamma=5e3)
    + dephasing_excited_states("D1", gamma=1e7)
)


def hamil(
        mw_det,
        b_longitudinal=0.1,
        mw_mag_field=1e-2,
        laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/5):

    # Atom-field Hamiltonian
    # sigma_plus
    ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()
    # ham_laser_atom[8:11, :] = 0
    # ham_laser_atom[:, 8:11] = 0
    ham_laser_atom[:3, :] = 0       # F=1 -> F' neglected
    ham_laser_atom[:, :3] = 0
    # Hyperfine Structure with Zeeman levels
    hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
    eigvals, eigstates = hb0.eigenstates()
    F_states_reordered = [
        eigstates[2],
        eigstates[1],
        eigstates[0],
    ]
    for k in range(3, 3 + 5):
        F_states_reordered.append(eigstates[k])
    # Atom Hamiltonian in rotating frame
    ham_atom = H_atom(det_Light=0, line="D1").full()
    ham_atom[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)
    ham_atom[8:, 8:] = hb0.transform(F_states_reordered).tidyup(
        atol=1e-3)/3   # for excited state: g'_F = g_F / 3
    diff_f2_fp1 = ham_atom[9, 9] - ham_atom[5, 5]
    for k in range(8, 16):
        # laser resonant to F=2 m_F=0 to F'=1, m'_F=0
        ham_atom[k, k] -= diff_f2_fp1

    diff_f2_f1 = (ham_atom[5, 5]-ham_atom[1, 1])
    for k in range(3):
        ham_atom[k, k] += diff_f2_f1  # rotating frame mw

    # for k in range(8, 11):
    #     ham_atom[k, k] = 0      # we ignore F'=1
    # for k in range(11, 16):
    #     # laser resonant to all Zeeman levels
    #     ham_atom[k, k] = ham_atom[k-8, k-8]
    hb_ac = H_B(bx=mw_mag_field/2**0.5, by=mw_mag_field/2 **
                0.5).transform(F_states_reordered).tidyup(atol=1e-3)  # transverse MW field
    hb_ac = hb_ac.full()
    for i in range(7):  # RWA
        hb_ac[i, i + 1] = 0.0
        hb_ac[i + 1, i] = 0.0
    h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
    h_a_mw[:8, :8] = hb_ac

    ham_tot = ham_atom + h_a_mw + ham_laser_atom
    for k in range(3):
        ham_tot[k, k] += mw_det
    offset = ham_tot[1, 1]
    for k in range(16):
        ham_tot[k, k] -= offset
    return Qobj(ham_tot).tidyup(atol=1e-3)


# %%
laser_intens = OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/10
b_longitudinal = 0.1
Bmw = 1e-3
ham_tot = hamil(0, b_longitudinal=b_longitudinal,
                laser_intens=laser_intens, mw_mag_field=Bmw)
laser_ss = steadystate(
    hamil(0, b_longitudinal=b_longitudinal,
          laser_intens=laser_intens, mw_mag_field=0),
    c_op_list=decays
)
mw_detunings = np.linspace(ham_tot[3, 3] - ham_tot[0, 0] - 500e3*2*pi,
                           ham_tot[7, 7] - ham_tot[2, 2] + 500e3*2*pi,
                           201)

# %%
steadystates_both = [
    steadystate(hamil(det, b_longitudinal, laser_intens=laser_intens),
                c_op_list=decays, method="direct")
    for det in mw_detunings
]
excited_state_pops = [sum(state.diag()[8:]) for state in steadystates_both]
plt.plot(mw_detunings/(2e3*pi), excited_state_pops)
plt.xlabel("mw detuning (kHz)")
plt.ylabel(r"$\rho_{ee}$")
resonant_mw_freqs = [b_longitudinal * 0.7e3 *
                     k for k in range(-3, 3+1)]   # 7 different transition frequencies
for elem in resonant_mw_freqs:
    plt.axvline(x=elem, color="tab:orange")
plt.tight_layout()

# %%
h = hamil(ham_tot[7, 7] - ham_tot[2, 2],
          b_longitudinal, laser_intens=laser_intens)
time_evo_options = Options(nsteps=2**5 * 1000)
res = mesolve(
    h,
    rho0=laser_ss,
    tlist=np.linspace(0, 1e-3, 5000),
    c_ops=decays,
    options=time_evo_options,
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)

# %%
h = hamil(ham_tot[3, 3] - ham_tot[0, 0],
          b_longitudinal, laser_intens=laser_intens)
res = mesolve(
    h,
    rho0=laser_ss,
    tlist=np.linspace(0, 1e-3, 5000),
    c_ops=decays,
    options=time_evo_options,
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)
# %%


# %%


def faraday_rot_angle(rho):
    wavelength_probe_laser = 780e-9
    density_atoms = 2.33e12 / (1e-2)**3
    length_cell = 2e-3
    detuning_probe = -30e9 * 2*pi
    return (
        sum([mF * (-1)**F * rho.matrix_element(get_ket_Fg_D1(F, mF).dag(), get_ket_Fg_D1(F, mF))
            for F in (1, 2) for mF in range(-F, F + 1)]).real
        * density_atoms*length_cell
        * wavelength_probe_laser**2
        * GAMMA_RAD_D2
        / detuning_probe
    )

# %%


def P_loop(b_mw, radius=0.7e-2, distance=0.03):
    return (
        b_mw * 1e-4
        * (constants.mu_0
           * radius**2
           / (2 * (distance**2 + radius**2) ** (3 / 2)))**(-1)
    )**2 * 50


# %%
Bmw = 1e-4
steadystates_both = [
    steadystate(hamil(det, b_longitudinal, laser_intens=laser_intens, mw_mag_field=Bmw),
                c_op_list=decays, method="direct")
    for det in mw_detunings
]
excited_state_pops = [sum(state.diag()[8:]) for state in steadystates_both]
plt.plot(mw_detunings/(2e3*pi), excited_state_pops)
plt.xlabel("mw detuning (kHz)")
plt.ylabel(r"$\rho_{ee}$")
resonant_mw_freqs = [b_longitudinal * 0.7e3 *
                     k for k in range(-3, 3+1)]   # 7 different transition frequencies
for elem in resonant_mw_freqs:
    plt.axvline(x=elem, color="tab:orange")
plt.tight_layout()

# %%
angles = [faraday_rot_angle(state) for state in steadystates_both]
plt.plot(mw_detunings/(2e3*pi), angles)


# %%

models = [LorentzianModel(prefix=f"p{ind}_") for ind in (1, 2, 3)]
mod = ConstantModel()
for m in models:
    mod = mod + m
pars = mod.make_params(
    p1_center=resonant_mw_freqs[0], p2_center=resonant_mw_freqs[2], p3_center=resonant_mw_freqs[4])
res = mod.fit(data=excited_state_pops, params=pars, x=mw_detunings/(2e3*pi))
res.plot(title=" ")
res
# %%
b_longitudinal=0.1
ham_tot = hamil(0, b_longitudinal=b_longitudinal,
                    laser_intens=laser_intens, mw_mag_field=0)
mw_detunings = np.linspace(ham_tot[3, 3] - ham_tot[0, 0] - 300e3*2*pi,
                            ham_tot[7, 7] - ham_tot[2, 2] + 300e3*2*pi,
                            201)
resonant_mw_freqs = [b_longitudinal * 0.7e3 *
                     k for k in range(-3, 3+1)]   # 7 different transition frequencies

def run_simulation_exp(
        mw_magnetic_field,
        laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/10,
        ):
    rho_steady_list = [
        steadystate(hamil(det, b_longitudinal, laser_intens=laser_intens, mw_mag_field=mw_magnetic_field),
                    c_op_list=decays, method="direct")
        for det in mw_detunings
    ]
    return rho_steady_list


def get_excited_state_pops(rho_list) -> List:
    return [sum(state.diag()[8:]) for state in rho_list]


def fit_simulation(excited_state_pops: List):
    from lmfit.models import LorentzianModel, ConstantModel
    models = [LorentzianModel(prefix=f"p{ind}_") for ind in (1, 2, 3)]
    mod = ConstantModel()
    for m in models:
        mod = mod + m
    pars = mod.make_params(p1_center=resonant_mw_freqs[0],
                           p2_center=resonant_mw_freqs[2],
                           p3_center=resonant_mw_freqs[4],
                           p1_amplitude=5.8635e-04,
                           p2_amplitude=5.5343e-04,
                           p3_amplitude=4.3310e-04,
                           p1_sigma=10.0427174,
                           p2_sigma=11.8743714,
                           p3_sigma=10.5537300,	
                           c=2.3210e-05)
    res = mod.fit(data=excited_state_pops,
                  params=pars, x=mw_detunings/(2e3*pi))
    return res

#%%
magnetic_fields = np.linspace(1e-5, 1e-4, 10)
spectra_steady = [run_simulation_exp(b) for b in magnetic_fields]
# spectra_steady = parallel_map(run_simulation_exp, magnetic_fields)
#%%
# import pickle
# with open("simulations_b_values_save", "wb") as fp:
#     pickle.dump(spectra_steady, fp)

# %%
# with open("simulations_b_values_save", "rb") as fp:
#     spectra_steady = pickle.load(spectra_steady, fp)
# %%

exc_states = [get_excited_state_pops(elem) for elem in spectra_steady]
#%%
fit_results = [fit_simulation(exc) for exc in exc_states]
# %%
d=dict([(key, []) for key in fit_results[-1].params.keys()])
for result in fit_results:
    for key in result.params.keys():
        d[key].append(result.params[key].value)
import pandas as pd
res_df = pd.DataFrame(d)
#%%
res_df[["p1_sigma", "p2_sigma", "p3_sigma"]].plot(subplots=False, style="o")
# %%
res_df[["p1_amplitude", "p2_amplitude", "p3_amplitude"]].plot(subplots=False, style="o")
# %%
for ind in range(len(exc_states)):
    plt.plot(exc_states[ind])
# %%

