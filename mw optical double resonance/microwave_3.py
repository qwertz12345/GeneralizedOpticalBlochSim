# %%
from init import *

b_longitudinal = 0.1
mw_mag_field = 1e-2
laser_intens = OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/20

decays = (
    natural_decay_ops_D1() + quenching_ops("D1") +
    F2_to_F1_ground_state_decay("D1", gamma=3000) +
    F1_to_F2_ground_state_decay("D1", gamma=3000) +
    dephasing_excited_states("D1", gamma=1e7)
    + dephasing_ground_states_D1(gamma=3000)
)

# Atom-field Hamiltonian
ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()  # sigma_plus
ham_laser_atom[8:11, :] = 0
ham_laser_atom[:, 8:11] = 0
ham_laser_atom[:3,:] = 0
ham_laser_atom[:,:3] = 0

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
ham_atom = np.zeros((16, 16), dtype=np.cdouble)
ham_atom[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)
diff = (ham_atom[5, 5]-ham_atom[1, 1])
for k in range(3):
    ham_atom[k, k] += diff  # rotating frame mw
offset = ham_atom[1, 1]
for k in range(16):
    ham_atom[k, k] -= offset
for k in range(8, 11):
    ham_atom[k, k] = 0      # we ignore F'=1
for k in range(11, 16):
    ham_atom[k, k] = ham_atom[k-8, k-8]     # laser resonant to all Zeeman levels

hb_ac = H_B(bx=mw_mag_field/2**0.5, by=mw_mag_field/2**0.5).transform(F_states_reordered).tidyup(atol=1e-3) # transverse MW field
hb_ac = hb_ac.full()
for i in range(7):  # RWA
    hb_ac[i, i + 1] = 0.0
    hb_ac[i + 1, i] = 0.0
h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
h_a_mw[:8, :8] = hb_ac
ham_tot = (Qobj(ham_atom) + Qobj(h_a_mw) + Qobj(ham_laser_atom)).tidyup(atol=1e-3)
ham_only_laser =  (Qobj(ham_atom) + Qobj(ham_laser_atom)).tidyup(atol=1e-3)
rho_ss_laser = steadystate(ham_only_laser, c_op_list=decays)
# %%
def hamil(det):
    ham_final = ham_tot.copy().full()
    for k in range(3, 8):
        ham_final[k, k] -= det
    for k in range(11, 16):
        ham_final[k, k] -= det
    ham_final = Qobj(ham_final)
    return ham_final

mw_detunings = np.linspace(ham_tot[3, 3] - ham_tot[0, 0] - 500e3*2*pi, 
ham_tot[7, 7] - ham_tot[2, 2] + 500e3*2*pi, 401)
excited_state_pops = [sum(steadystate(hamil(det), c_op_list=decays).diag()[11:]) for det in mw_detunings]
#%%
plt.plot(mw_detunings/(2e3*pi), excited_state_pops)
plt.xlabel("mw detuning (kHz)")
plt.ylabel(r"$\rho_{ee}$")
resonant_mw_freqs = [b_longitudinal * 0.7e3 *
                     k for k in range(-3, 3+1)]   # 7 different transition frequencies
for elem in resonant_mw_freqs:
    plt.axvline(x=elem, color="tab:orange")
plt.tight_layout()
#%%
mw_detuning = ham_tot[3, 3] - ham_tot[0, 0]     # detuning from clock transition
ham_final = hamil(mw_detuning)
res = mesolve(
    ham_final,
    rho0=rho_ss_laser,
    tlist=np.linspace(0, 1e-4, 1000),
    c_ops=decays,
    options=Options(nsteps=2**6 * 1000),
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)
# %%
rho_ss_both = steadystate(ham_final, c_op_list=decays)
maplot(rho_ss_both)
# %%
rho_ss_both.diag()[:8]-rho_ss_laser.diag()[:8]
# %%
