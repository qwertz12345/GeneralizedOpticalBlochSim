# %%
from init import *

# %%
def get_pump_intensity(power):
    return power / ((2.3e-3) ** 2 * pi)


def get_probe_intensity(power):
    return power / ((1.953e-3) ** 2 * pi)


# %%
print(f"{rabi_D1_vector_component(-1, E_0_plus(get_pump_intensity(1e-3))):.1e}")

# %% [markdown]
# Test if we get a quarter of the population in the excited state for the saturation intensity.

# %%
# Test if we get a quarter of the population in the excited state for the saturation intensity.
hamil = laser_sigma_plus_cycling_D2(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING)
liou = liouvillian(hamil, c_ops=natural_decay_ops_D2())
rho_ss = steadystate(liou)
rho_ss.matrix_element(get_ket_Fe_D2(3, 3).dag(), get_ket_Fe_D2(3, 3))

# %%
res = mesolve(liou, tlist=np.linspace(0, 1e-5, 200), rho0=sum([basis(24, i).proj() for i in range(3,8)]).unit(),
              options=Options(nsteps=17000))
# maplot(res.states[-1])
plot_excited_states_time(res)
plot_ground_states_time(res)

# %% [markdown]
# Test if detuning leads to correct excited state population. We expect 1/12.

# %%
# Test if detuning leads to correct excited state population. We expect 1/12.
hamil = laser_sigma_plus_cycling_D2(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING, detuning=GAMMA_RAD_D2)
liou = liouvillian(hamil, c_ops=natural_decay_ops_D2())
rho_ss = steadystate(liou)
rho_ss.matrix_element(get_ket_Fe_D2(3, 3).dag(), get_ket_Fe_D2(3, 3))

# %%
hamil = laser_sigma_plus_cycling_D2(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING)
liou = liouvillian(hamil, c_ops=natural_decay_ops_D2()+quenching_ops("D2")+F2_to_F1_ground_state_decay("D2")+intra_F_ground_decay("D2"))
rho_ss = steadystate(liou)
rho_ss.matrix_element(get_ket_Fe_D2(3, 3).dag(), get_ket_Fe_D2(3, 3))

# %%
rho_ss = steadystate(liou, method="eigen")
rho_ss.matrix_element(get_ket_Fe_D2(3, 3).dag(), get_ket_Fe_D2(3, 3))

# %% [markdown]
# # MW
# 
# 

# %% [markdown]
# ## Test simple case of 2 level system

# %%
# test simple case of 2 level system
hamil = H_mw(0, 0, 0.01)
# , c_ops=natural_decay_ops_D1()+[quenching_ops("D1")]+[F2_to_F1_ground_state_decay("D1")]+[intra_F_ground_decay("D1")])
liou = liouvillian(hamil)
res = mesolve(liou, tlist=np.linspace(0, 1e-3, 1000),
              rho0=get_ket_Fg_D1(2, 0).proj(),
              options=Options(nsteps=1000))
# maplot(res.states[-1])
# plot_excited_states_time(res)
plot_ground_states_time(res)

# %% [markdown]
# ## Test simple case of 2 level system, off resonant

# %%
# test simple case of 2 level system off resonant
hamil = H_mw(0, 0, 0.01, det_mw=10e3)
# , c_ops=natural_decay_ops_D1()+[quenching_ops("D1")]+[F2_to_F1_ground_state_decay("D1")]+[intra_F_ground_decay("D1")])
liou = liouvillian(hamil)
res = mesolve(liou, tlist=np.linspace(0, 1e-3, 1000),
              rho0=get_ket_Fg_D1(2, 0).proj(),
              options=Options(nsteps=1000))
# maplot(res.states[-1])
# plot_excited_states_time(res)
plot_ground_states_time(res)

# %%

# 2nd Laser
# def H_af_2nd_laser(omega_p, rabi_vector, line):
#     left = 1/2 * sum([np.conjugate(rabi_vector[ind]) * sigma_q(q, line=line)
#                       for ind, q in enumerate((1, -1, 0))])
#     right = left.dag()
#     args = {"omega_p": omega_p}
#     return [[left+right, "cos(omega_p * t)"], [1j*(left-right), "sin(omega_p * t)"]]



# %% [markdown]
# # spectroscopy

# %%
# spectroscopy
def get_equally_ground_state_D1():
    return sum([basis(16, i).proj() for i in range(8)]).unit()


def projector_excited_D1():
    return sum([basis(16, k).proj() for k in range(8, 16)])



# %%
rt = mesolve(liouvillian(laser_sigma_plus_F2_FP1_D1(1e-3, det=0),c_ops=natural_decay_ops_D1()+quenching_ops("D1")), tlist=np.linspace(0, 1e-6, 200), options=Options(nsteps=4000), rho0=get_equally_ground_state_D1())
plot_excited_states_time(rt)


# %%
rt = mesolve(liouvillian(laser_sigma_plus_F2_FP1_D1(1e-3, det=1e9*2*pi),c_ops=natural_decay_ops_D1()+quenching_ops("D1")), tlist=np.linspace(0, 1e-6, 200), options=Options(nsteps=4000), rho0=get_equally_ground_state_D1())
plot_excited_states_time(rt)


# %%
# def spec_solve(freq):
#     try:
#         steps = 1000
#         res_spec = mesolve(laser_sigma_plus_F2_FP1_D1(1e-3, det=freq),
#                         rho0=get_equally_ground_state_D1(),
#                         tlist=np.linspace(0, 1e-6, 100),
#                         c_ops=natural_decay_ops_D1(),
#                         e_ops=[projector_excited_D1()],
#                         options=Options(nsteps=steps))
#         return res_spec.expect[0][-1]
#     except Exception:
#         steps = 4000
#         try:
#             res_spec = mesolve(laser_sigma_plus_F2_FP1_D1(1e-3, det=freq),
#                         rho0=get_equally_ground_state_D1(),
#                         tlist=np.linspace(0, 1e-6, 100),
#                         c_ops=natural_decay_ops_D1(),
#                         e_ops=[projector_excited_D1()],
#                         options=Options(nsteps=steps))
#             return res_spec.expect[0][-1]
#         except Exception:
#             steps = 8000
#             res_spec = mesolve(laser_sigma_plus_F2_FP1_D1(1e-3, det=freq),
#                         rho0=get_equally_ground_state_D1(),
#                         tlist=np.linspace(0, 1e-6, 100),
#                         c_ops=natural_decay_ops_D1(),
#                         e_ops=[projector_excited_D1()],
#                         options=Options(nsteps=steps))
#             return res_spec.expect[0][-1]


def spec_solve(freq):
    steps = 5000
    res_spec = mesolve(laser_sigma_plus_F2_FP1_D1(1e-3, det=freq),
                       rho0=get_equally_ground_state_D1(),
                       tlist=np.linspace(0, 1e-6, 100),
                       c_ops=natural_decay_ops_D1(),
                       e_ops=[projector_excited_D1()],
                       options=Options(nsteps=steps))
    return res_spec.expect[0][-1]



# %%


# %%
laser_freq_scan = np.linspace(-2e8*2*pi, 1e9*2*pi, 200)
results = [spec_solve(f) for f in laser_freq_scan]
plt.plot(laser_freq_scan, results)

# %%
laser_freq_scan = np.linspace(-2e8*2*pi, 1e9*2*pi, 200)
results = [spec_solve1(f) for f in laser_freq_scan]
plt.plot(laser_freq_scan, results)

# %%
from spec_parallel import spec_solve1
laser_freq_scan = np.linspace(-2e8*2*pi, 1e9*2*pi, 2000)
r = parallel_map(spec_solve1,  laser_freq_scan)
plt.plot(laser_freq_scan, r)

# %%



