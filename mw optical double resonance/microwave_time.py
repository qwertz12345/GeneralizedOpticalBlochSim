# %%
from init import *
# %%
b_longitudinal = 0.1
resonant_mw_freqs = [
    b_longitudinal * 0.7e6 * 2 * pi * k for k in range(-3, 3 + 1)
]  # 7 transition frequencies
decays = (natural_decay_ops_D1() +
          F2_to_F1_ground_state_decay("D1", gamma=3000) +
          F1_to_F2_ground_state_decay("D1", gamma=3000))
laser_intensity = OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL
ham_laser = laser_sigma_plus_F2_FP2_D1(laser_intensity)
# %%
hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)  # in uncoupled basis!
eigvals, eigstates = hb0.eigenstates()
F_states_reordered = [
    eigstates[2],
    eigstates[1],
    eigstates[0],
]  # reorder the eigenstates to have same basis in the same order: |F=1, m=-1>, |F=1, m=0>, ...
for k in range(3, 3 + 5):
    F_states_reordered.append(eigstates[k])
eigvals_reordered = [eigvals[2], eigvals[1], eigvals[0]]
for k in range(3, 3 + 5):
    eigvals_reordered.append(eigvals[k])
# %%
magnetic_field_mw = B_loop(10)/2**0.5
hb_ac = H_B(bx=magnetic_field_mw, by=magnetic_field_mw,
            bz=0).transform(F_states_reordered)
hb_ac = hb_ac.full()
for i in range(7):  # RWA
    hb_ac[i, i + 1] = 0.0
    hb_ac[i + 1, i] = 0.0
dims = 16
h_a_mw = np.zeros(shape=(dims, dims), dtype=np.cdouble)
h_a_mw[:8, :8] = hb_ac
hamil_mw = Qobj(h_a_mw)

hamil_b = np.zeros(shape=(dims, dims), dtype=np.cdouble)
hamil_b[:8, :8] = H_B(bz=b_longitudinal).transform(F_states_reordered)
hamil_b = Qobj(hamil_b)

# %%
total_hamil = [ham_laser+hamil_b, [hamil_mw, "cos(omega * t + phase)"]]
res = mesolve(
    total_hamil,
    rho0=steadystate(ham_laser, c_op_list=decays),
    c_ops=decays,
    tlist=np.linspace(0, 1e-9, 2000),
    options=Options(nsteps=2**5*1000),
    progress_bar=True,
    args={"omega": (total_hamil[0].diag()[3]-total_hamil[0].diag()[2]), "phase": 0.0},
)
qsave(res, "laser and mw with decay")
plot_excited_states_time(res)
plot_ground_states_time(res)


# %%
total_hamil = [H_atom(0, "D1")+hamil_b, [hamil_mw, "cos(omega * t + phase)"]]
res = mesolve(
    total_hamil,
    rho0=get_ket_Fg_D1(1, -1).proj(),
    # c_ops=decays,
    tlist=np.linspace(0, 1e-5, 2000),
    options=Options(nsteps=2**8*1000),
    progress_bar=True,
    args={"omega": abs(total_hamil[0].diag()[3]-total_hamil[0].diag()[0]), "phase": 0.0},
)
grnd_pops = [[elem.diag()[k].real for elem in res.states] for k in range(8)]
plt.plot(res.times, grnd_pops[0])
#%%
# qsave(res, "laser and mw with decay")
plot_excited_states_time(res)
plot_ground_states_time(res)

# %%
