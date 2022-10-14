#%%
from lib import *

h = H_atom(0, "D2")
h


# %%
# laser_freq = (get_ket_Fe_D2(3,3).dag()*h*get_ket_Fe_D2(3,3) - get_ket_Fg_D2(2,2).dag()*h*get_ket_Fg_D2(2,2))[0,0]
# Hyperfine Structure with Zeeman levels
b_longitudinal = 1
hb0 = H_hfs_ground() + H_B(bz=b_longitudinal)
eigvals, eigstates = hb0.eigenstates()
F_states_reordered = [
    eigstates[2],
    eigstates[1],
    eigstates[0],
]
for k in range(3, 3 + 5):
    F_states_reordered.append(eigstates[k])
h = H_atom(0, "D2").full()
det = h[-1, -1] - h[7, 7]
h = H_atom(det - 20e9 * 2 * pi, "D2").full()
h[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3).full()
shift = h[1, 1]
for k in range(24):
    h[k, k] = h[k, k] - shift
h = Qobj(h)
h_total = (
    h
    + H_atom_field_D2(-1, E_0_plus(250e-6 / (pi * (50e-6) ** 2) / 2))
    + H_atom_field_D2(1, E_0_plus(250e-6 / (pi * (50e-6) ** 2) / 2))
)
#%%
rho_ss = steadystate(h_total, c_op_list=natural_decay_ops_D2())
maplot(rho_ss)
plot_bar_ground_pop(rho_ss)
# %%
time_evo_options = Options(nsteps=2**10 * 1000)
times = np.linspace(0, 2e-3, 50000)
res = mesolve(
    h_total,
    rho0=get_ket_Fg_D2(2, 2).proj(),
    tlist=times,
    c_ops=natural_decay_ops_D2(),
    options=time_evo_options,
    progress_bar=True,
)
plot_excited_states_time(res)
plot_ground_states_time(res)
#%%
plt.plot([rho.tr() for rho in res.states])
# %%
expect = [abs(rho[7, 7]) for rho in res.states]
plt.plot(times, expect)
# %%
expect = [abs(rho[6, 6]) for rho in res.states]
plt.plot(times, expect)
# %%
expect = [abs(rho[2, 2]) for rho in res.states]
plt.plot(times, expect)
#%%
qsave(res, "result_minus20ghz")