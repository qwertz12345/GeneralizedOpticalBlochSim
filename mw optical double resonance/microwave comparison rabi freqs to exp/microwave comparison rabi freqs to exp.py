# # %% [markdown]
# # # Rabi Oscillations 

# # %%
# from lmfit.models import LorentzianModel, ConstantModel
# from genobs.lib import *
# decays = (
#     natural_decay_ops_D1() 
#     # + quenching_ops("D1") 
#     + wall_coll("D1", gamma=1e3)
#     # + dephasing_excited_states("D1", gamma=1e7)
#     # + dephasing_ground_states_D1()
# )


# def hamil(
#         mw_det,
#         b_longitudinal=0.1,
#         mw_mag_field=1e-2,
#         laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/5):

#     # Atom-field Hamiltonian
#     # sigma_plus
#     ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()
#     # ham_laser_atom[8:11, :] = 0
#     # ham_laser_atom[:, 8:11] = 0
#     ham_laser_atom[:3, :] = 0       # F=1 -> F' neglected
#     ham_laser_atom[:, :3] = 0
#     ham_laser_atom[:, -5:] = 0
#     ham_laser_atom[-5:, :] = 0
#     # Hyperfine Structure with Zeeman levels
#     hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
#     eigvals, eigstates = hb0.eigenstates()
#     F_states_reordered = [
#         eigstates[2],
#         eigstates[1],
#         eigstates[0],
#     ]
#     for k in range(3, 3 + 5):
#         F_states_reordered.append(eigstates[k])
#     # Atom Hamiltonian in rotating frame
#     ham_atom = H_atom(det_Light=0, line="D1").full()
#     ham_atom[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)
#     ham_atom[8:, 8:] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)/3   # for excited state: g'_F = g_F / 3

#     diff_f2_fp1 = ham_atom[9, 9] - ham_atom[5, 5]  # laser resonant to F=2 m_F=-2 to F'=1, m'_F=-1
#     for k in range(8, 16):
#         ham_atom[k, k] -= diff_f2_fp1

#     diff_f2_f1 = (ham_atom[5, 5]-ham_atom[1, 1])
#     for k in range(3):
#         ham_atom[k, k] += diff_f2_f1  # rotating frame mw

#     # for k in range(8, 11):
#     #     ham_atom[k, k] = 0      # we ignore F'=1
#     # for k in range(11, 16):
#     #     # laser resonant to all Zeeman levels
#     #     ham_atom[k, k] = ham_atom[k-8, k-8]
#     hb_ac = H_B(
#         bx=mw_mag_field, 
#         by=0
#         ).transform(F_states_reordered).tidyup(atol=1e-3)  # transverse MW field
#     hb_ac = hb_ac.full()
#     for i in range(7):  # RWA
#         hb_ac[i, i + 1] = 0.0
#         hb_ac[i + 1, i] = 0.0
#     h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
#     h_a_mw[:8, :8] = hb_ac
#     ham_tot = ham_atom + h_a_mw + ham_laser_atom
#     for k in range(3):
#         ham_tot[k, k] += mw_det
#     offset = ham_tot[1, 1]
#     for k in range(16):
#         ham_tot[k, k] -= offset
#     return Qobj(ham_tot).tidyup(atol=1e-3)


# def P_loop(b_mw, radius=0.7e-2, distance=0.03):
#     return (
#         b_mw * 1e-4
#         * (constants.mu_0
#             * radius**2
#             / (2 * (distance**2 + radius**2) ** (3 / 2)))**(-1)
#     )**2 * 50

# # %% [markdown]
# # ## Laser Intensity as Parameter
# # ### 1

# # %%
# laser_intens = 0.89 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### Quenching On

# # %%
# from lmfit.models import LorentzianModel, ConstantModel
# from init import *
# decays = (
#     natural_decay_ops_D1() +
#     quenching_ops("D1") +
#     wall_coll("D1", gamma=1e3)
#     # + dephasing_excited_states("D1", gamma=1e7) +
#     # dephasing_ground_states_D1()
# )


# def hamil(
#         mw_det,
#         b_longitudinal=0.1,
#         mw_mag_field=1e-2,
#         laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/5):

#     # Atom-field Hamiltonian
#     # sigma_plus
#     ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()
#     # ham_laser_atom[8:11, :] = 0
#     # ham_laser_atom[:, 8:11] = 0
#     ham_laser_atom[:3, :] = 0       # F=1 -> F' neglected
#     ham_laser_atom[:, :3] = 0
#     ham_laser_atom[:, -5:] = 0
#     ham_laser_atom[-5:, :] = 0
#     # Hyperfine Structure with Zeeman levels
#     hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
#     eigvals, eigstates = hb0.eigenstates()
#     F_states_reordered = [
#         eigstates[2],
#         eigstates[1],
#         eigstates[0],
#     ]
#     for k in range(3, 3 + 5):
#         F_states_reordered.append(eigstates[k])
#     # Atom Hamiltonian in rotating frame
#     ham_atom = H_atom(det_Light=0, line="D1").full()
#     ham_atom[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)
#     ham_atom[8:, 8:] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)/3   # for excited state: g'_F = g_F / 3

#     diff_f2_fp1 = ham_atom[9, 9] - ham_atom[5, 5]  # laser resonant to F=2 m_F=-2 to F'=1, m'_F=-1
#     for k in range(8, 16):
#         ham_atom[k, k] -= diff_f2_fp1

#     diff_f2_f1 = (ham_atom[5, 5]-ham_atom[1, 1])
#     for k in range(3):
#         ham_atom[k, k] += diff_f2_f1  # rotating frame mw

#     # for k in range(8, 11):
#     #     ham_atom[k, k] = 0      # we ignore F'=1
#     # for k in range(11, 16):
#     #     # laser resonant to all Zeeman levels
#     #     ham_atom[k, k] = ham_atom[k-8, k-8]
#     hb_ac = H_B(
#         bx=mw_mag_field, 
#         by=0
#         ).transform(F_states_reordered).tidyup(atol=1e-3)  # transverse MW field
#     hb_ac = hb_ac.full()
#     for i in range(7):  # RWA
#         hb_ac[i, i + 1] = 0.0
#         hb_ac[i + 1, i] = 0.0
#     h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
#     h_a_mw[:8, :8] = hb_ac
#     ham_tot = ham_atom + h_a_mw + ham_laser_atom
#     for k in range(3):
#         ham_tot[k, k] += mw_det
#     offset = ham_tot[1, 1]
#     for k in range(16):
#         ham_tot[k, k] -= offset
#     return Qobj(ham_tot).tidyup(atol=1e-3)


# def P_loop(b_mw, radius=0.7e-2, distance=0.03):
#     return (
#         b_mw * 1e-4
#         * (constants.mu_0
#             * radius**2
#             / (2 * (distance**2 + radius**2) ** (3 / 2)))**(-1)
#     )**2 * 50

# # %%
# laser_intens = 0.89 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### 2 (even weaker laser)

# # %%
# laser_intens = 0.89 * 10e-1     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### another laser intens

# # %%
# laser_intens = 0.89 * 10e-0     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### and another laser intens

# # %%
# laser_intens = 0.5 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### and another

# # %%
# laser_intens = 0.25 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %% [markdown]
# # ### another....

# # %%
# from lmfit.models import LorentzianModel, ConstantModel
# from init import *
# decays = (
#     natural_decay_ops_D1() 
#     + quenching_ops("D1") 
#     + wall_coll("D1", gamma=1e3)
#     # + dephasing_excited_states("D1", gamma=1e7)
#     # + dephasing_ground_states_D1()
# )


# def hamil(
#         mw_det,
#         b_longitudinal=0.1,
#         mw_mag_field=1e-2,
#         laser_intens=OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL/5):

#     # Atom-field Hamiltonian
#     # sigma_plus
#     ham_laser_atom = H_atom_field_D1(-1, E_0_plus(laser_intens)).full()
#     # ham_laser_atom[8:11, :] = 0
#     # ham_laser_atom[:, 8:11] = 0
#     ham_laser_atom[:3, :] = 0       # F=1 -> F' neglected
#     ham_laser_atom[:, :3] = 0
#     ham_laser_atom[:, -5:] = 0
#     ham_laser_atom[-5:, :] = 0
#     # Hyperfine Structure with Zeeman levels
#     hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
#     eigvals, eigstates = hb0.eigenstates()
#     F_states_reordered = [
#         eigstates[2],
#         eigstates[1],
#         eigstates[0],
#     ]
#     for k in range(3, 3 + 5):
#         F_states_reordered.append(eigstates[k])
#     # Atom Hamiltonian in rotating frame
#     ham_atom = H_atom(det_Light=0, line="D1").full()
#     ham_atom[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)
#     ham_atom[8:, 8:] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)/3   # for excited state: g'_F = g_F / 3

#     diff_f2_fp1 = ham_atom[9, 9] - ham_atom[5, 5]  # laser resonant to F=2 m_F=-2 to F'=1, m'_F=-1
#     for k in range(8, 16):
#         ham_atom[k, k] -= diff_f2_fp1

#     diff_f2_f1 = (ham_atom[5, 5]-ham_atom[1, 1])
#     for k in range(3):
#         ham_atom[k, k] += diff_f2_f1  # rotating frame mw

#     # for k in range(8, 11):
#     #     ham_atom[k, k] = 0      # we ignore F'=1
#     # for k in range(11, 16):
#     #     # laser resonant to all Zeeman levels
#     #     ham_atom[k, k] = ham_atom[k-8, k-8]
#     hb_ac = H_B(
#         bx=mw_mag_field, 
#         by=0
#         ).transform(F_states_reordered).tidyup(atol=1e-3)  # transverse MW field
#     hb_ac = hb_ac.full()
#     for i in range(7):  # RWA
#         hb_ac[i, i + 1] = 0.0
#         hb_ac[i + 1, i] = 0.0
#     h_a_mw = np.zeros(shape=(16, 16), dtype=np.cdouble)
#     h_a_mw[:8, :8] = hb_ac
#     ham_tot = ham_atom + h_a_mw + ham_laser_atom
#     for k in range(3):
#         ham_tot[k, k] += mw_det
#     offset = ham_tot[1, 1]
#     for k in range(16):
#         ham_tot[k, k] -= offset
#     return Qobj(ham_tot).tidyup(atol=1e-3)


# def P_loop(b_mw, radius=0.7e-2, distance=0.03):
#     return (
#         b_mw * 1e-4
#         * (constants.mu_0
#             * radius**2
#             / (2 * (distance**2 + radius**2) ** (3 / 2)))**(-1)
#     )**2 * 50

# # %%
# laser_intens = 0.25 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-2
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 4e-4, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# g=np.abs(exc_states)

# # %%
# plt.plot(g)

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %%
# from lmfit.model import Model
# import numpy as np
# def rabi_osci_fctn(t, A, B, C, steady, gamma1, gamma2, rabi):
#     # C = 1 - A - steady                                   # we set  f(t=0) = 1
#     # B = (gamma1 * A + gamma2 * C) / rabi                 # we set f'(t=0) = 0
#     return (
#         steady
#         + A * np.exp(- gamma1 * t)
#         + B * np.sin(rabi * t) * np.exp(- gamma2 * t)
#         + C * np.cos(rabi * t) * np.exp(- gamma2 * t)
#     )
# from copy import copy


# # %%
# y = g
# t = res.times
# mod = Model(rabi_osci_fctn, independent_vars=["t"])
# pars = mod.make_params()
# # try:
# #     pars = copy(resfit.params)
# # except NameError:
# pars["gamma1"].set(value=3e3)#, min=1e3, max=99e3)
# pars["gamma2"].set(value=22e3)#, min=1.000e3, max=66e3)
# pars["rabi"].set(value=55e3)#, min=0.01e3, max=77e3)
# # pars["A"].set(min=0, max=0.6, value=0.05)
# # pars["B"].set(min=0, max=0.6, value=0.01)
# # pars["C"].set(min=0, max=0.6, value=0.01)
# pars["A"].set(value=0.05e-5)
# pars["B"].set(value=0.1e-5)
# pars["C"].set(value=0.1e-5)

# pars["steady"].set(value=y[-1])#, min=0.8, max=0.999)


# resfit = mod.fit(
#     data=y,
#     t=t,
#     params=pars,
#     # method="differential_evolution",
# )

# # %%
# resfit.plot()

# # %%
# resfit

# # %% [markdown]
# # ### weaker MW

# # %%
# laser_intens = 0.01 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 1e-3
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 1e-3, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# g=np.abs(exc_states)

# # %%
# plt.plot(g)

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %%
# from lmfit.model import Model
# import numpy as np
# def rabi_osci_fctn(t, A, B, C, steady, gamma1, gamma2, rabi):
#     # C = 1 - A - steady                                   # we set  f(t=0) = 1
#     # B = (gamma1 * A + gamma2 * C) / rabi                 # we set f'(t=0) = 0
#     return (
#         steady
#         + A * np.exp(- gamma1 * t)
#         + B * np.sin(rabi * t) * np.exp(- gamma2 * t)
#         + C * np.cos(rabi * t) * np.exp(- gamma2 * t)
#     )
# from copy import copy


# # %%
# y = g
# t = res.times
# mod = Model(rabi_osci_fctn, independent_vars=["t"])
# pars = mod.make_params()
# # try:
# #     pars = copy(resfit.params)
# # except NameError:
# pars["gamma1"].set(value=5e3)#, min=1e3, max=99e3)
# pars["gamma2"].set(value=5e3)#, min=1.000e3, max=66e3)
# pars["rabi"].set(value=22e3, min=0.1e3, max=77e3)
# # pars["A"].set(min=0, max=0.6, value=0.05)
# # pars["B"].set(min=0, max=0.6, value=0.01)
# # pars["C"].set(min=0, max=0.6, value=0.01)
# pars["A"].set(value=0.05e-5)
# pars["B"].set(value=0.1e-5)
# pars["C"].set(value=0.1e-5)
# pars["steady"].set(value=y[-1])#, min=0.8, max=0.999)


# resfit = mod.fit(
#     data=y,
#     t=t,
#     params=pars,
#     # method="differential_evolution",
# )

# # %%
# resfit.plot();

# # %%
# resfit

# # %%
# h

# # %%
# y[-1]/y[0]

# # %%


# # %%
# laser_intens = 0.01 * 10     # W/m²
# b_longitudinal = 41 / 700    # G
# Bmw = 2e-3
# ham_clock = hamil(0, b_longitudinal=b_longitudinal,
#                 laser_intens=laser_intens, mw_mag_field=0)
# laser_ss = steadystate(
#     ham_clock,
#     c_op_list=decays
# )
# plot_bar_excited_pop_D1(laser_ss)
# plot_bar_ground_pop(laser_ss)
# mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
# h = hamil(mw_detuning,
#           b_longitudinal, 
#           laser_intens=laser_intens, 
#           mw_mag_field=Bmw)

# # %%
# mw_ss = steadystate(h, c_op_list=decays)
# plot_bar_excited_pop_D1(mw_ss-laser_ss)
# plot_bar_ground_pop(mw_ss-laser_ss)

# # %%
# time_evo_options = Options(nsteps=2**5 * 1000)
# res = mesolve(
#     h,
#     rho0=laser_ss,
#     tlist=np.linspace(0, 1e-3, 2000),
#     c_ops=decays,
#     options=time_evo_options,
#     progress_bar=True
# )
# plot_excited_states_time(res)
# plot_ground_states_time(res)

# # %%
# exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
# plt.plot(res.times, exc_states)#/exc_states[0])

# # %%
# g=np.abs(exc_states)

# # %%
# plt.plot(g)

# # %%
# f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
# plt.plot(res.times, f2_states)#/f2_states[0])

# # %%
# from lmfit.model import Model
# import numpy as np
# def rabi_osci_fctn(t, A, B, C, steady, gamma1, gamma2, rabi):
#     # C = 1 - A - steady                                   # we set  f(t=0) = 1
#     # B = (gamma1 * A + gamma2 * C) / rabi                 # we set f'(t=0) = 0
#     return (
#         steady
#         + A * np.exp(- gamma1 * t)
#         + B * np.sin(rabi * t) * np.exp(- gamma2 * t)
#         + C * np.cos(rabi * t) * np.exp(- gamma2 * t)
#     )
# from copy import copy


# # %%
# y = g
# t = res.times
# mod = Model(rabi_osci_fctn, independent_vars=["t"])
# pars = mod.make_params()
# # try:
# #     pars = copy(resfit.params)
# # except NameError:
# pars["gamma1"].set(value=5e3)#, min=1e3, max=99e3)
# pars["gamma2"].set(value=5e3)#, min=1.000e3, max=66e3)
# pars["rabi"].set(value=22e3, min=0.1e3, max=77e3)
# # pars["A"].set(min=0, max=0.6, value=0.05)
# # pars["B"].set(min=0, max=0.6, value=0.01)
# # pars["C"].set(min=0, max=0.6, value=0.01)
# pars["A"].set(value=0.05e-5)
# pars["B"].set(value=0.1e-5)
# pars["C"].set(value=0.1e-5)
# pars["steady"].set(value=y[-1])#, min=0.8, max=0.999)


# resfit = mod.fit(
#     data=y,
#     t=t,
#     params=pars,
#     # method="differential_evolution",
# )

# # %%
# resfit.plot();

# # %%
# resfit

# # %%
# h

# # %%
# from scipy import constants
# constants.physical_constants["mag. constant"][0] / (2*constants.pi * 0.02) * (10**(35/20) * 63e-3 /50) * 1e4
 

# # %%
# qutip.settings.atol

# %%
#%%
from genobs.lib import *
decays = (
    natural_decay_ops_D1() 
    # + quenching_ops("D1") 
    + wall_coll("D1", gamma=1e3)
    # + dephasing_excited_states("D1", gamma=1e7)
    # + dephasing_ground_states_D1()
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
    ham_laser_atom[:, -5:] = 0
    ham_laser_atom[-5:, :] = 0
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
    ham_atom[8:, 8:] = hb0.transform(F_states_reordered).tidyup(atol=1e-3)/3   # for excited state: g'_F = g_F / 3

    diff_f2_fp1 = ham_atom[9, 9] - ham_atom[5, 5]  # laser resonant to F=2 m_F=-2 to F'=1, m'_F=-1
    for k in range(8, 16):
        ham_atom[k, k] -= diff_f2_fp1

    diff_f2_f1 = (ham_atom[5, 5]-ham_atom[1, 1])
    for k in range(3):
        ham_atom[k, k] += diff_f2_f1  # rotating frame mw

    # for k in range(8, 11):
    #     ham_atom[k, k] = 0      # we ignore F'=1
    # for k in range(11, 16):
    #     # laser resonant to all Zeeman levels
    #     ham_atom[k, k] = ham_atom[k-8, k-8]
    hb_ac = H_B(
        bx=mw_mag_field, 
        by=0
        ).transform(F_states_reordered).tidyup(atol=1e-3)  # transverse MW field
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


def P_loop(b_mw, radius=0.7e-2, distance=0.03):
    return (
        b_mw * 1e-4
        * (constants.mu_0
            * radius**2
            / (2 * (distance**2 + radius**2) ** (3 / 2)))**(-1)
    )**2 * 50

# %% [markdown]
# ## Laser Intensity as Parameter
# ### 1

# %%
laser_intens = 0.89 * 10     # W/m²
b_longitudinal = 41 / 700    # G
Bmw = 1e-2
ham_clock = hamil(0, b_longitudinal=b_longitudinal,
                laser_intens=laser_intens, mw_mag_field=0)
laser_ss = steadystate(
    ham_clock,
    c_op_list=decays
)
plot_bar_excited_pop_D1(laser_ss)
plot_bar_ground_pop(laser_ss)
mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
h = hamil(mw_detuning,
          b_longitudinal, 
          laser_intens=laser_intens, 
          mw_mag_field=Bmw)

# %%
mw_ss = steadystate(h, c_op_list=decays)
plot_bar_excited_pop_D1(mw_ss-laser_ss)
plot_bar_ground_pop(mw_ss-laser_ss)

# %%
time_evo_options = Options(nsteps=2**5 * 1000)
res = mesolve(
    h,
    rho0=laser_ss,
    tlist=np.linspace(0, 4e-4, 2000),
    c_ops=decays,
    options=time_evo_options,
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)
sim_results = []
fit_results = []
import numpy as np
laser_intens = 0.01 * 10     # W/m²
b_longitudinal = 0.1    # G
ham_clock = hamil(0, b_longitudinal=b_longitudinal,
                laser_intens=laser_intens, mw_mag_field=0)
mw_detuning = (ham_clock[3, 3] - ham_clock[0, 0])
from lmfit.model import Model
def rabi_osci_fctn(t, A, B, C, steady, gamma1, gamma2, rabi):
        # C = 1 - A - steady                                   # we set  f(t=0) = 1
        # B = (gamma1 * A + gamma2 * C) / rabi                 # we set f'(t=0) = 0
        return (
            steady
            + A * np.exp(- gamma1 * t)
            + B * np.sin(rabi * t) * np.exp(- gamma2 * t)
            + C * np.cos(rabi * t) * np.exp(- gamma2 * t)
        )
from copy import copy

def sim(Bmw):
    laser_ss = steadystate(
        ham_clock,
        c_op_list=decays
    )
    # plot_bar_excited_pop_D1(laser_ss)
    # plot_bar_ground_pop(laser_ss)
    h = hamil(mw_detuning,
            b_longitudinal, 
            laser_intens=laser_intens, 
            mw_mag_field=Bmw)   
    mw_ss = steadystate(h, c_op_list=decays)
    # plot_bar_excited_pop_D1(mw_ss-laser_ss)
    # plot_bar_ground_pop(mw_ss-laser_ss)
    time_evo_options = Options(nsteps=2**5 * 1000)
    res = mesolve(
        h,
        rho0=laser_ss,
        tlist=np.linspace(0, 1e-3, 2000),
        c_ops=decays,
        options=time_evo_options,
        progress_bar=True
    )
    # plot_excited_states_time(res)
    # plot_ground_states_time(res)
    exc_states = np.array([sum(state.diag()[8:]) for state in res.states])
    # plt.plot(res.times, exc_states)#/exc_states[0])
    g=np.abs(exc_states)
    # plt.plot(g)
    f2_states = np.array([sum(state.diag()[3:8]) for state in res.states])
    # plt.plot(res.times, f2_states)#/f2_states[0])
    y = g
    t = res.times
    mod = Model(rabi_osci_fctn, independent_vars=["t"])
    pars = mod.make_params()
    # try:
    #     pars = copy(resfit.params)
    # except NameError:
    pars["gamma1"].set(value=15e3, min=1e3, max=55e3)
    pars["gamma2"].set(value=30e3, min=1.000e3, max=66e3)
    pars["rabi"].set(value=77e3, min=1e3, max=177e3)
    # pars["A"].set(min=0, max=0.6, value=0.05)
    # pars["B"].set(min=0, max=0.6, value=0.01)
    # pars["C"].set(min=0, max=0.6, value=0.01)
    pars["A"].set(value=5e-5)
    pars["B"].set(value=-4e-5)
    pars["C"].set(value=10e-5)
    pars["steady"].set(value=y[-1])#, min=0.8, max=0.999)
    resfit = mod.fit(
        data=y,
        t=t,
        params=pars,
        # method="differential_evolution",
    )
    return res, resfit


Bs = np.linspace(1e-3, 1e-2, 10)
for B in Bs:
    a, b = sim(B)
    sim_results.append(a)
    fit_results.append(b)

# %%
from lmfit.model import Model
def rabi_osci_fctn(t, A, B, C, steady, gamma1, gamma2, rabi):
    # C = 1 - A - steady                                   # we set  f(t=0) = 1
    # B = (gamma1 * A + gamma2 * C) / rabi                 # we set f'(t=0) = 0
    return (
        steady
        + A * np.exp(- gamma1 * t)
        + B * np.sin(rabi * t) * np.exp(- gamma2 * t)
        + C * np.cos(rabi * t) * np.exp(- gamma2 * t)
    )
newfit=[]
for s,f in zip(sim_results, fit_results):

    exc_states = np.array([sum(state.diag()[8:]) for state in s.states])
    # plt.plot(res.times, exc_states)#/exc_states[0])

    g=np.abs(exc_states)
    y = g
    t = s.times
    mod = Model(rabi_osci_fctn, independent_vars=["t"])
    pars = mod.make_params()
    pars = f.params
    pars["rabi"].set(min=3e3, max=177e3, value=150e3)
    pars["steady"].set(value=y[-1])#, min=0.8, max=0.999)

    
    resfit = mod.fit(
        data=y,
        t=t,
        params=pars,
        # method="nelder",
    )
    newfit.append(resfit)
    resfit.plot()
# %%
