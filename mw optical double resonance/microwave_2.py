# %%
from init import *

g_F2 = 2.00233113 * (2*(2+1) - 3/2*(3/2+1) + 1/2*(1/2+1)) / (2*2*(2+1))

b_longitudinal = 0.1
mw_mag_field = 1e-3
laser_intens = OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/10

resonant_mw_freqs = [b_longitudinal * 0.7e6 * 2 * pi *
                     k for k in range(-3, 3+1)]   # 7 different transition frequencies
decays = (
    natural_decay_ops_D1() + quenching_ops("D1") +
    F2_to_F1_ground_state_decay("D1", gamma=3000) +
    F1_to_F2_ground_state_decay("D1", gamma=3000) +
    dephasing_excited_states("D1", gamma=1e7)
    + dephasing_ground_states_D1(gamma=3000)
)
ham_only_laser = laser_sigma_plus_F2_FP2_D1(laser_intens)
rho_ss_laser = steadystate(ham_only_laser, c_op_list=decays)

# atom in static magnetic, longitudinal field: 0.1 Gauss
hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
eigvals, eigstates = hb0.eigenstates()
# reorder the eigenstates to have same basis in the same order: |F=1, m=-1>, |F=1, m=0>, ...
F_states_reordered = [
    eigstates[2],
    eigstates[1],
    eigstates[0],
]
for k in range(3, 3 + 5):
    F_states_reordered.append(eigstates[k])

ham_atom = H_atom(0, "D1").full()
laser_freq = (ham_atom[-1, -1] - ham_atom[3, 3])  # D1 F2 -> F'2
for k in range(8, 16):      # rotating frame
    ham_atom[k, k] -= laser_freq
ham_atom[:8, :8] = hb0.transform(
    F_states_reordered).tidyup(atol=1e-5).full()
diff = (ham_atom[5, 5]-ham_atom[1, 1])
for k in range(3):
    ham_atom[k, k] += diff  # rotating frame mw
offset = ham_atom[1, 1]
for k in range(16):
    ham_atom[k, k] -= offset
    
hb_ac = H_B(bx=mw_mag_field/2**0.5, by=mw_mag_field/2 **
            0.5).transform(F_states_reordered).tidyup(atol=1e-5)
hb_ac = hb_ac.full()
for i in range(7):  # RWA
    hb_ac[i, i + 1] = 0.0
    hb_ac[i + 1, i] = 0.0
h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
h_a_mw[:8, :8] = hb_ac
ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens))  # sigma_plus
ham_tot = ham_laser_atom + Qobj(h_a_mw) + Qobj(ham_atom)
# ham_tot_cleaned = ham_tot.copy().full()
# ham_tot_cleaned[-5:] = 0
# ham_tot = Qobj(ham_tot_cleaned)
# %%
res = mesolve(
    ham_tot,
    rho0=rho_ss_laser,
    tlist=np.linspace(0, 1e-5, 1000),
    c_ops=decays,
    options=Options(nsteps=2**6 * 1000),
    progress_bar=True
)
# %%
plot_excited_states_time(res)
plot_ground_states_time(res)
# %%
rho_ss_both = steadystate(ham_tot, c_op_list=decays)
maplot(rho_ss_both)
# %%
rho_ss_both.diag()[:8]-rho_ss_laser.diag()[:8]
# %%


# %%
def ham_mw_det(det):
    tmp = ham_tot.copy().full()
    for i in range(3, 8):
        tmp[i, i] = ham_tot[i, i] - det
    return Qobj(tmp)


# %%
mw_detunings = np.linspace(resonant_mw_freqs[0]-111e3*2*pi,
                           resonant_mw_freqs[-1]+111e3*2*pi,
                           1001)
ground_pops = [sum(steadystate(ham_mw_det(det), c_op_list=decays).diag()[:8])
               for det in mw_detunings]
plt.plot(mw_detunings/(2e3*pi), ground_pops)
# %%
