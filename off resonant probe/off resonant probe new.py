#%%

from qutip import *
from genobs.Rubidium87_operators import *
import numpy as np
import matplotlib.pyplot as plt
from visualizations import *
#%%
b_longitudinal = 1
beam_intens = calculate_beam_intensity(250e-6, 50e-6)
haf = H_atom_field_D2(1, E_0_plus(beam_intens/2)) + H_atom_field_D2(-1, E_0_plus(beam_intens/2))
h = H_atom(0, "D2") + haf + H_B(bz=b_longitudinal, line="D2")
freq_highest_transition = (
    h.matrix_element(ket_Fe_D2(3, 3).dag(), ket_Fe_D2(3, 3))
    - h.matrix_element(ket_Fg_D2(1, 1).dag(), ket_Fg_D2(1, 1))
).real
freq_lowest_transition = (
    h.matrix_element(ket_Fe_D2(0, 0).dag(), ket_Fe_D2(0, 0)) 
    - h.matrix_element(ket_Fg_D2(2, 2).dag(), ket_Fg_D2(2, 2))
).real

h_det = np.zeros((24, 24), dtype=np.cdouble)
for k in range(16):
    h_det[k+8, k+8] = 2e9*2*pi + freq_highest_transition
h_det = Qobj(h_det)
# %%
h = h - h_det
# %%
rho_ss = steadystate(h, c_op_list=natural_decay_ops_D2())
maplot(rho_ss)
# %%

time_evo_options = Options(nsteps=2**10 * 1000)
times = np.linspace(0, 2e-5, 10000)
res = mesolve(
    h,
    rho0=ket_Fg_D2(2, 2).proj(),
    tlist=times,
    c_ops=natural_decay_ops_D2(),
    options=time_evo_options,
    progress_bar=True
)
plot_excited_states_time(res)
plot_ground_states_time(res)
#%%
plt.plot([rho.tr() for rho in res.states])
plt.title("trace of rho")
# %%
expect = [abs(rho[7, 7]) for rho in res.states]
plt.plot(times, expect)
# %%
expect = [abs(rho[6, 6]) for rho in res.states]
plt.plot(times, expect)
# %%
expect = [abs(rho[2, 2]) for rho in res.states]
plt.plot(times, expect)
# %%
