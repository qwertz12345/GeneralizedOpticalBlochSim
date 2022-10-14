#%%
from qutip import *
import numpy as np
from genobs.Rubidium87_operators import *
from genobs.visualizations import *
import matplotlib.pyplot as plt

#%%
B_z = 1e4
h0d1 = H_atom(0, "D1") + H_B("D1", bz=B_z).tidyup(atol=1e-3)
h0d1_eigenvals, h0d1_eigenstates = h0d1.eigenstates()
hd1 = h0d1 + H_atom_field_D1(-1, E_0_plus(OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL))
order_new_basis = [0, 1, 2, -4, -3, -2, -1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
hd1_transformed = hd1.transform(
    [h0d1_eigenstates[k] for k in order_new_basis]).tidyup(
    1e-3
)
laserd1_freq = hd1_transformed[8,8] - hd1_transformed[0,0]
hd1_transformed = hd1_transformed.full()

for k in range(8):
    hd1_transformed[k+8, k+8] -= laserd1_freq
#%%
h0d2 = H_atom(0, "D2") + H_B("D2", bz=B_z)
h0d2_eigenvals, h0d2_eigenstates = h0d2.eigenstates()
hd2 = h0d2 + H_atom_field_D2(
    -1, E_0_plus(SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING / 1000)
)
order_new_basis1 = (
    [4, 5, 6]
    + list(range(-8, -4))
    +[7]
    + [0, 1, 2, 3]
    + list(range(8, 16))
    + [-4, -3, -2, -1]
)
hd2_transformed = hd2.transform(
    [h0d2_eigenstates[k] for k in order_new_basis1]).tidyup(
    1e-3
)
laserd2_freq = hd2_transformed[8,8] - hd2_transformed[3,3] 
hd2_transformed = hd2_transformed.full()
for k in range(8, 24):
    hd2_transformed[k, k] -= laserd2_freq

#%%
# combine to one object
H = np.zeros((32, 32), dtype=np.cdouble)
H[:16, :16] = hd1_transformed[:, :]
H[16:, :8] = hd2_transformed[8:, :8]
H[:8, 16:] = hd2_transformed[:8, 8:]
H[16:, 16:] = hd2_transformed[8:, 8:]
H = Qobj(H)
# %%
decays = []
for dec in natural_decay_ops_D1():
    dec_transformed = dec.transform([h0d1_eigenstates[k] for k in order_new_basis])
    d = np.zeros((32, 32), dtype=np.cdouble)
    d[:16, :16] = dec_transformed[:, :]
    decays.append(d)
for newdec, dec in zip(decays, natural_decay_ops_D2()):
    dec_transformed = dec.transform([h0d2_eigenstates[k] for k in order_new_basis1])
    newdec[:8, 16:] = dec_transformed[:8, 8:]
for n, ar in enumerate(decays):
    decays[n] = Qobj(ar)

# %%
rho_ss = steadystate(H, decays)
# %%
matrixplot(rho_ss)
# %%
plot_bar_ground_pop(rho_ss)
#%%
res = mesolve(
    H,
    c_ops=decays,
    tlist=np.linspace(0, 1e-4, 2000),
    options=Options(nsteps=10**6),
    rho0=basis(32, 7).proj(),
    progress_bar=True
)
# %%
plt.plot(res.times, [state[0, 0] for state in res.states])
# %%
plot_ground_states_time(res)
# %%
plt.plot(res.times, [state.tr() for state in res.states])
# %%
