# %%
from init import *
# %% [markdown]
# # Setting: D1, Laser resonant to F=1 $\rightarrow$ F', $\sigma_+$
#
# Intensity:  1 mW

# %%
get_pump_intensity(1e-3)  # W/m^2 = 1000 mW/(100 cm)^2 = 0.1 mW/cm^2

# %%
hamil = laser_sigma_plus_F2_FP2_D1(get_pump_intensity(1e-3))
rabi_D1_vector_component(E_0_plus(get_pump_intensity(1e-3)))/(1e6*2*pi)

# %%
fig, ax = maplot(hamil, annot=True)
fig.set_size_inches(14, 7)


# %% [markdown]
# # Radiative Decay

# %%
maplot(sum(natural_decay_ops_D1()))

# %% [markdown]
# ## Steady State

# %%
L = liouvillian(hamil, c_ops=natural_decay_ops_D1())

# %%
rho_ss = steadystate(L)
maplot(rho_ss)


# %%
res_r = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=8000),
)
_, axe = plot_excited_states_time(res_r)
_, axg = plot_ground_states_time(res_r)

# %% [markdown]
# ## Quenching added

# %%
L = liouvillian(hamil, c_ops=natural_decay_ops_D1() + quenching_ops("D1"))
rho_ss = steadystate(L)
maplot(rho_ss)

# %%
res_q = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=8000),
)
plot_excited_states_time(res_q)
plot_ground_states_time(res_q)

# %% [markdown]
# ## ground state decay

# %% [markdown]
# ## no drive, intra F decay

# %%
L = liouvillian(
    None,
    c_ops=intra_F_ground_decay("D1"),
)

res_g = mesolve(
    L,
    rho0=get_ket_Fg_D1(2, 1).proj(),
    tlist=np.linspace(0, 1e-3, 400),
    options=Options(nsteps=5000),
)
plot_ground_states_time(res_g)


# %% [markdown]
# ## F=2 to F=1

# %%
L = liouvillian(
    None,
    c_ops=F2_to_F1_ground_state_decay("D1"),
)
res_wall = mesolve(
    L,
    rho0=get_ket_Fg_D1(2, 1).proj(),
    tlist=np.linspace(0, 1e-3, 400),
    options=Options(nsteps=5000),
)
plot_ground_states_time(res_wall)


# %% [markdown]
# ## F=1 to F=2

# %%
L = liouvillian(
    None,
    c_ops=F1_to_F2_ground_state_decay("D1"),
)
res_wall = mesolve(
    L,
    rho0=get_ket_Fg_D1(1, 1).proj(),
    tlist=np.linspace(0, 1e-3, 400),
    options=Options(nsteps=5000),
)
plot_ground_states_time(res_wall)


# %% [markdown]
# ## combined
#

# %%
L = liouvillian(
    None,
    c_ops=F1_to_F2_ground_state_decay(
        "D1")+F2_to_F1_ground_state_decay("D1")+intra_F_ground_decay("D1"),
)
res_comb = mesolve(
    L,
    rho0=get_ket_Fg_D1(2, 1).proj(),
    tlist=np.linspace(0, 1e-2, 400),
    options=Options(nsteps=5000),
)
plot_ground_states_time(res_comb)


# %%
res_comb.states[-1]

# %% [markdown]
# ## alternative: collaps operators acting on electron spin alone

# %%
c_ops_ground = []
sts = f_ground_states_uncoupled()
for sigma_k in jmat(1/2):
    c = tensor(identity(4), sigma_k).transform(sts[1])
    c.dims = [[8], [8]]
    tmp = np.zeros(shape=(16, 16), dtype=np.cdouble)
    tmp[:8, :8] = c
    c_ops_ground.append((2e3)**(1/2)*Qobj(tmp))

# %%
c_ops_ground[2]

# %%
L = liouvillian(
    None,
    c_ops=c_ops_ground
)
res_spin = mesolve(
    L,
    rho0=get_ket_Fg_D1(2, 1).proj(),
    tlist=np.linspace(0, 1e-2, 400),
    options=Options(nsteps=5000),
)
plot_ground_states_time(res_spin)

# %% [markdown]
# ## difference

# %%
maplot(res_q.states[-1] - res_r.states[-1])

# %%
maplot(res_g.states[-1] - res_q.states[-1])

# %%
maplot(res_g.states[-1] - res_r.states[-1])

# %% [markdown]
# # everything added

# %%
H_atom(0, "D1").eigenenergies()

# %%
1.91912041e+09+3.19853402e+09

# %%
hamil = laser_sigma_plus_F2_FP2_D1(get_pump_intensity(1e-3))

L = liouvillian(
    hamil,
    # F1_to_F2_ground_state_decay("D1")+F2_to_F1_ground_state_decay("D1")+intra_F_ground_decay("D1")+quenching_ops("D1"),
    c_ops=natural_decay_ops_D1()
)
res_all = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=7000),
)
plot_excited_states_time(res_all)
plot_ground_states_time(res_all)


# %%
L = liouvillian(
    hamil,
    c_ops=natural_decay_ops_D1() + F1_to_F2_ground_state_decay("D1") +
    F2_to_F1_ground_state_decay(
        "D1")+intra_F_ground_decay("D1")+quenching_ops("D1"),
)
res_all = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=7000),
)
plot_excited_states_time(res_all)
plot_ground_states_time(res_all)


# %%

L = liouvillian(hamil, c_ops=natural_decay_ops_D1() +
                dephasing_excited_states("D1"))

rho_ss = steadystate(L)
maplot(rho_ss)
res_deph = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=8000),
)
_, axe = plot_excited_states_time(res_deph)
_, axg = plot_ground_states_time(res_deph)


# %%
L = liouvillian(hamil, c_ops=natural_decay_ops_D1())

rho_ss = steadystate(L)
maplot(rho_ss)
res_deph = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=8000),
)
#%%
_, axe = plot_excited_states_time(res_deph)
_, axg = plot_ground_states_time(res_deph)


# %%
L = liouvillian(hamil, c_ops=natural_decay_ops_D1() +
                dephasing_excited_states("D1", gamma=1e9))

rho_ss = steadystate(L)
maplot(rho_ss)
res_deph = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=8000),
)
fige, axde = plot_excited_states_time(res_deph, axe)
figg, axdg = plot_ground_states_time(res_deph, axg)


# %%
L = liouvillian(
    hamil,
    c_ops=natural_decay_ops_D1() + F1_to_F2_ground_state_decay("D1") +
    F2_to_F1_ground_state_decay("D1")+intra_F_ground_decay("D1")+quenching_ops("D1") +
    dephasing_excited_states("D1"),
)
res_all = mesolve(
    L,
    rho0=get_equally_ground_state_D1(),
    tlist=np.linspace(0, 1e-5, 400),
    options=Options(nsteps=7000),
)
#%%
fige, axe = plot_excited_states_time(res_all, axs=axe)
figg, axg = plot_ground_states_time(res_all, axs=axg)

# %%
rho_ss = steadystate(L)
maplot(rho_ss)
# %%
