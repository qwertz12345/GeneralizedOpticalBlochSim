# %%
from init import *

def H_mw_corrected(Bmw_x, Bmw_y, Bmw_z, line="D1", det_mw=0.0, b_static_z=0.08): # not tested for D2
    """Total Hamiltonian for atom-MW interaction (no laser).
    Detuning from clock transition"""
    hb0 = H_hfs_ground() + H_B(bz=b_static_z)  # atom in static magnetic, longitudinal field
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

    energ_shifted = [
        en - eigvals_reordered[1] for en in eigvals_reordered
    ]  # shift energies so that |F=1, mF=0> corresponds to 0   
    ens = [
        en if k < 3 else en - energ_shifted[5]   # rotating frame
        for k, en in enumerate(energ_shifted)
    ]
    energ_shifted1 = np.array(ens) + H_atom(0, line="D1")[3, 3] # shift 0 to lowest energy in hamiltonian of atom
    # ens = eigvals_reordered
    # det_mw -= 
    h_a = sum([(energ_shifted1[m + 2 + 3] - det_mw) * ket_Fg(2, m).proj() for m in range(-2, 2 + 1)]) + sum([energ_shifted1[m + 1] * ket_Fg(1, m).proj() for m in range(-1, 2)])
    dims = 16 if line == "D1" else 24
    hb_ac = H_B(bx=Bmw_x, by=Bmw_y, bz=Bmw_z).transform(F_states_reordered)
    hb_ac = hb_ac.full()
    for i in range(7):  # RWA
        hb_ac[i, i + 1] = 0.0
        hb_ac[i + 1, i] = 0.0
    h_a_mw = np.zeros(shape=(dims, dims), dtype=np.cdouble)
    h_a_mw[:8, :8] = h_a
    h_a_mw[:8, :8] = hb_ac + h_a_mw[:8, :8]
    return Qobj(h_a_mw)
# %%
b_longitudinal = 0.1
# htemp = H_mw(0,0,0, b_static_z=b_longitudinal)

# resonant_mw_freqs = [
#     htemp.eigenenergies()[0]-htemp.eigenenergies()[-2],
#     htemp.eigenenergies()[0],
#     htemp.eigenenergies()[0] - htemp.eigenenergies()[1],
#     htemp.eigenenergies()[3],
#     htemp.eigenenergies()[-2],
#     htemp.eigenenergies()[-1],
#     htemp.eigenenergies()[-1]+htemp.eigenenergies()[-2]
# ]
resonant_mw_freqs = [b_longitudinal * 0.7e6 * 2 * pi * k for k in range(-3, 3+1)]   # 7 transition frequencies
decays = (
    natural_decay_ops_D1() + quenching_ops("D1") +
    F2_to_F1_ground_state_decay("D1", gamma=3000) +
    F1_to_F2_ground_state_decay("D1", gamma=3000) +
    dephasing_excited_states("D1")
    + dephasing_ground_states_D1(gamma=3000)
)
laser_intensity = SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING / 10
ham_laser = laser_sigma_plus_F2_FP2_D1(laser_intensity)
tmp = ham_laser.copy().full()
tmp[:3, :] = 0
tmp[:, :3] = 0
tmp[8:11, :8] = 0
tmp[:8, 8:11] = 0
for k in range(8):
    tmp[k,k] = 0 
hamil_approx = Qobj(tmp)
ham_laser = hamil_approx + H_mw_corrected(0,0,0)
rho_ss_laser = steadystate(ham_laser, c_op_list=decays)
magnetic_field_mw = B_loop(0.1)
rabi_mw_est = MU_BOHR * magnetic_field_mw   # without transition amplitude -> rough estimate
times = np.linspace(0, 1/rabi_mw_est * 10, 1000)
ham_mw = H_mw_corrected(
    magnetic_field_mw/2**0.5, magnetic_field_mw/2**0.5, 0, 
    det_mw=resonant_mw_freqs[0], 
    b_static_z=b_longitudinal)
# %%
ham_laser
# %%
ham_mw

# %%
rabi_laser = rabi_D1_vector_component(E_0_plus(laser_intensity))
f"{rabi_D1_vector_component(E_0_plus(laser_intensity)): .2e}"

#%%
f"{rabi_mw_est:.2e}"
# %% [markdown]
# ## Time Evo due to Laser
# %%
res_laser = mesolve(ham_laser,
                    tlist=np.linspace(0, 3*2*pi/rabi_laser, 1000),
                    rho0=get_equally_ground_state_D1(),
                    options=Options(nsteps=2**3 * 1000),
                    progress_bar=True)
# qsave(res_laser, "res_laser")                          
plot_excited_states_time(res_laser)
plot_ground_states_time(res_laser)
# %%
res_laser_decay = mesolve(ham_laser,
                          tlist=np.linspace(0, 6*2*pi/rabi_laser, 1000),
                          rho0=get_equally_ground_state_D1(),
                          options=Options(nsteps=2**3 * 1000),
                          c_ops=decays,
                          progress_bar=True)
# qsave(res_laser_decay, "res_laser_decay")
#%%
# res_laser_decay = qload("res_laser_decay")
plot_excited_states_time(res_laser_decay)
plot_ground_states_time(res_laser_decay)


# %%
maplot(res_laser_decay.states[-1])


# %%
maplot(rho_ss_laser)

# %% [markdown]
# ## Time Evo of resonant MW transitions without Decays

# %%
time_evo_res_all_transitions = []
for ind, freq in enumerate(resonant_mw_freqs):
    result = mesolve(
        H_mw_corrected(magnetic_field_mw/2**0.5, magnetic_field_mw/2**0.5, 0, det_mw=freq, b_static_z=b_longitudinal),
        tlist=times,
        rho0=(get_ket_Fg_D1(1, -1).proj()+get_ket_Fg_D1(1, 0).proj()+get_ket_Fg_D1(1, 1).proj())/3,
        options=Options(nsteps=2**1 * 1000)
    )
    plot_ground_states_time(result)
    time_evo_res_all_transitions.append(result)
# %%
for elem in time_evo_res_all_transitions:
    maplot(elem.states[-1])

# %%
res_laser_decay1 = mesolve(ham_laser,
                          tlist=np.linspace(0, 1e-6, 1001),
                          rho0=rho_ss_laser,
                          options=Options(nsteps=2**1 * 1000),
                          c_ops=decays,
                          progress_bar=True)
plot_excited_states_time(res_laser_decay1)
plot_ground_states_time(res_laser_decay1)
maplot(res_laser_decay1.states[-1])

# %% [markdown]
# ## MW Evo with Decay

# %% [markdown]
# ### First MW transition

# %%
res_mw_decay_1 = mesolve(ham_mw,
                    tlist=times,
                    rho0=get_ket_Fg_D1(1, -1).proj(),
                    options=Options(nsteps=2**4 * 1000),
                    c_ops=decays,
                    progress_bar=True)
# qsave(res_mw_decay_1, "res_mw_decay_1")
plot_excited_states_time(res_mw_decay_1)
plot_ground_states_time(res_mw_decay_1)


# %% [markdown]
# ### Third MW transition ($\pi$)

# %%
res_mw_decay_3 = mesolve(H_mw_corrected(
    5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[2], b_static_z=b_longitudinal),
                       tlist=np.linspace(0, 1e-3, 1001),
                                           rho0=get_ket_Fg_D1(1, -1).proj(),
                       options=Options(nsteps=2**3 * 1000),
                       c_ops=decays,
                       progress_bar=True)
# qsave(res_mw_decay_3, "res_mw_decay_3")
plot_excited_states_time(res_mw_decay_3)
plot_ground_states_time(res_mw_decay_3)


# %%
maplot(res_mw_decay.states[-1])
rho_ss_mw = steadystate(ham_mw, c_op_list=decays)
maplot(rho_ss_mw)

# %% [markdown]
# ## Laser and MW Time Evo
# ### MW Transition 1

# %%
# times = np.linspace(0, 1e-3, 1001)
res_both_decay_1 = mesolve(ham_mw + ham_laser,
                         tlist=times,
                         rho0=rho_ss_laser,
                         options=Options(nsteps=2**6 * 1000),
                         c_ops=decays,
                         progress_bar=True)
# qsave(res_both_decay_1, "res_both_decay_1")
#%%
# res_both_decay_1 =  qload("res_both_decay_1")
plot_excited_states_time(res_both_decay_1)
plot_ground_states_time(res_both_decay_1)
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_1.states]
plt.figure()
plt.plot(res_both_decay_1.times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
# res_both_decay = qload("res_both_decay")


# %%
sum(res_both_decay_1.states[-1].diag()[8:])

# %%
sum(rho_ss_laser.diag()[8:])

# %%
(sum(res_both_decay_1.states[-1].diag()[8:]) - sum(rho_ss_laser.diag()[8:])) / sum(rho_ss_laser.diag()[8:])

# %% [markdown]
# ### MW resonant to transition 3 (double transition)

# %%
ham = ham_laser + H_mw(
    magnetic_field_mw/2**0.5, magnetic_field_mw/2**0.5, 0, 
    det_mw=resonant_mw_freqs[2], b_static_z=b_longitudinal)
res_both_decay_t3 = mesolve(ham,
    tlist=times,
    rho0=rho_ss_laser,
    options=Options(nsteps=2**6 * 1000),
    c_ops=decays,
    progress_bar=True)
qsave(res_both_decay_t3, "res_both_decay_t3")
#%%

plot_excited_states_time(res_both_decay_t3)
plot_ground_states_time(res_both_decay_t3)
# maplot(res_both_decay.states[-1])
# rho_ss_both = steadystate(ham, c_op_list=decays)
# maplot(rho_ss_both)
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_t3.states]
plt.figure()
plt.plot(res_both_decay_t3.times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
# res_both_decay_t3 = qload("res_both_decay_t3")


# %%
sum(res_both_decay_t3.states[-1].diag()[8:])

# %%
sum(rho_ss_laser.diag()[8:])

# %%
(sum(res_both_decay_t3.states[-1].diag()[8:]) - sum(rho_ss_laser.diag()[8:])) / sum(rho_ss_laser.diag()[8:])

# %% [markdown]
# ### mw resonant to pi transition 
# 
# There should be no resonance since $B_{mw} \bf{e}_z = 0 $.

# %%
ham = ham_laser + H_mw(
    magnetic_field_mw/2**0.5, magnetic_field_mw/2**0.5, 0, 
    det_mw=resonant_mw_freqs[1], b_static_z=b_longitudinal)
res_both_decay_pi = mesolve(ham,
                        tlist=times,
                        rho0=rho_ss_laser,
                        options=Options(nsteps=2**8 * 1000),
                        c_ops=decays,
                        progress_bar=True)
qsave(res_both_decay_pi, "res_both_decay_pi")
plot_excited_states_time(res_both_decay_pi)
plot_ground_states_time(res_both_decay_pi)
# maplot(res_both_decay_pi.states[-1])
rho_ss_both = steadystate(ham, c_op_list=decays)
# maplot(rho_ss_both)
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_pi.states]

# %%
plt.figure()
plt.plot(np.linspace(0, 5e-5, 1001), rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
sum(res_both_decay_pi.states[-1].diag()[8:])

# %%
sum(rho_ss_laser.diag()[8:])

# %%
sum(res_both_decay_pi.states[-1].diag()[8:]) - sum(rho_ss_laser.diag()[8:])

# %% [markdown]
# ### Clock Transition
# 
# Should be off-resonant, like all $\pi$ transitions

# %%
ham = (ham_laser + H_mw(5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[3], b_static_z=b_longitudinal))
times = np.linspace(0, 1e-3, 1001)
res_both_decay_clock = mesolve(ham,
                         tlist=times,
                         rho0=rho_ss_laser,
                         options=Options(nsteps=2**6 * 1000),
                         c_ops=decays,
                         progress_bar=True)
plot_excited_states_time(res_both_decay_clock)
plot_ground_states_time(res_both_decay_clock)
qsave(res_both_decay_clock, "res_both_decay_clock")
# maplot(res_both_decay_clock.states[-1])
rho_ss_both = steadystate(ham, c_op_list=decays)
# maplot(rho_ss_both)


# %%
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_clock.states]
plt.figure()
plt.plot(times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
sum(rho_ss_laser.diag()[8:])

# %%
sum(res_both_decay_clock.states[-1].diag()[8:])

# %%
sum(rho_ss_laser.diag()[8:])-sum(res_both_decay_clock.states[-1].diag()[8:])

# %% [markdown]
# ### Comparison $\rho_{ee}$

# %%
# res_both_decay_clock = qload("res_both_decay_clock")
# res_both_decay_pi = qload("res_both_decay_pi")
# res_both_decay_t3 = qload("res_both_decay_t3")
# res_mw_decay_1 = qload("res_mw_decay_1")

# %%
times = np.linspace(0, 1e-3, 1001)

plt.figure()
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_clock.states], label="clock 4")
plt.plot(times,[sum(state.diag()[8:]) for state in res_both_decay_pi.states], label="pi 1")
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_t3.states], label="double 2")
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_1.states], label="sigma-, transition 1")
plt.legend()
plt.ylabel(r"$\rho_{ee}$")

# %%
plt.figure()
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_clock.states], "1", label="clock 4")
# plt.plot(times,[sum(state.diag()[8:]) for state in res_both_decay_pi.states], label="pi 1")
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_t3.states], "2", label="double 2")
plt.plot(times, [sum(state.diag()[8:]) for state in res_both_decay_1.states], "3", label="sigma-, transition 1")
plt.xlim(-1e-6, 3e-5)
plt.legend()
plt.ylabel(r"$\rho_{ee}$")

# %%
ham = (laser_sigma_plus_F2_FP2_D1(get_pump_intensity(10e-6)) 
    + H_mw(5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[3], b_static_z=b_longitudinal))
times = np.linspace(0, 5e-4, 1001)
res_both_decay_clock = mesolve(ham,
                         tlist=times,
                         rho0=rho_ss_laser,
                         options=Options(nsteps=2**5 * 1000),
                         c_ops=decays,
                         progress_bar=True)
plot_excited_states_time(res_both_decay_clock)
plot_ground_states_time(res_both_decay_clock)
maplot(res_both_decay_clock.states[-1])
rho_ss_both = steadystate(ham, c_op_list=decays)
maplot(rho_ss_both)
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_clock.states]
plt.figure()
plt.plot(times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
(rho_ss_both-rho_ss_laser).diag()

# %%
sum((rho_ss_both-rho_ss_laser).diag()[8:])

# %%
sum((rho_ss_both-rho_ss_laser).diag()[:8])

# %%
maplot(rho_ss_both-rho_ss_laser)

# %%
H_mw(5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[3], b_static_z=b_longitudinal).eigenenergies()

# %%
resonant_mw_freqs[3]

# %%
eigens, eigvecs = H_mw(5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[3], b_static_z=b_longitudinal).eigenstates()

# %%
ham = (laser_sigma_plus_F2_FP2_D1(get_pump_intensity(10e-6)) 
    + H_mw(5e-4, 5e-4, 0, det_mw=resonant_mw_freqs[-1], b_static_z=b_longitudinal))
times = np.linspace(0, 5e-5, 1001)
res_both_decay_last = mesolve(ham,
                         tlist=times,
                         rho0=rho_ss_laser,
                         options=Options(nsteps=2**3 * 1000),
                         c_ops=decays,
                         progress_bar=True)
plot_excited_states_time(res_both_decay_last)
plot_ground_states_time(res_both_decay_last)
maplot(res_both_decay_last.states[-1])
rho_ss_both = steadystate(ham, c_op_list=decays)
maplot(rho_ss_both)
rho_ee_t = [sum(state.diag()[8:]) for state in res_both_decay_last.states]
plt.figure()
plt.plot(times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")

# %%
Qobj(sum(decays).full()[:8,:8])

# %%
b_longitudinal = 0.1
resonant_mw_freqs = [b_longitudinal * 0.7e6 * 2 * pi * k for k in range(-3, 3+1)]
mw_freq = resonant_mw_freqs[0]
decays = (natural_decay_ops_D1() + quenching_ops("D1") +
          F2_to_F1_ground_state_decay("D1", gamma=3000) +
          F1_to_F2_ground_state_decay("D1", gamma=3000) +
          dephasing_excited_states("D1"))

# %%
Qobj(sum(decays).full()[:8,:8])

# %%
dephasing_excited_states("D1")

# %%
dephasing_excited_states("D1")[0]

# %%
Qobj(sum(decays).full()[8:,8:])

# %%
b_longitudinal = 0.1
resonant_mw_freqs = [b_longitudinal * 0.7e6 * 2 * pi * k for k in range(-3, 3+1)]
mw_freq = resonant_mw_freqs[0]
decays = (natural_decay_ops_D1() + quenching_ops("D1") +
          F2_to_F1_ground_state_decay("D1", gamma=3000) +
          F1_to_F2_ground_state_decay("D1", gamma=3000) +
          dephasing_excited_states("D1"))

# %%
Qobj(sum(decays).full()[8:,8:])

# %%
sum(decays)

# %%
Qobj(sum(decays).full()[8:,:])

# %%
Qobj(sum(decays).full()[8:,:8])

# %%
Qobj(sum(decays).full()[:8,8:])


