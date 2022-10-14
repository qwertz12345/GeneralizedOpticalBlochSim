#%%
from genobs.lib import *
h = H_atom(0, "D2")
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
# h = H_atom(0*2*pi, "D2").full()
h = h.full()
h[:8, :8] = hb0.transform(F_states_reordered).tidyup(atol=1e-3).full()

#%%
shift = h[1, 1]
for k in range(24):
    h[k, k] = h[k, k] - shift
h = Qobj(h)
h_total = h + H_atom_field_D2(-1, E_0_plus(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING)) + H_atom_field_D2(1, E_0_plus(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING))
h_total /= 1e6
h_total
# %%
f11_fp33 = h_total[-1, -1] - h_total[2, 2]
# %%
h_total = h_total.full()
detuning = 1000 # MHz
for k in range(8, 24):
    h_total[k, k] += detuning*2*pi + f11_fp33
h_total = Qobj(h_total)
# %%
time_evo_options = Options(nsteps=2**5 * 1000)
times = np.linspace(0, 10, 5000)
res = mesolve(
    h_total,
    rho0=get_ket_Fg_D2(2, 2).proj(),
    tlist=times,
    c_ops=natural_decay_ops_D2(),
    options=time_evo_options,
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)
# %%
