# %%
# import plotly.express as px
# about()
from genobs.lib import *

# %%
h = H_atom_field_D1(-1, E_0_plus(10)) + H_atom(0, "D1")
h
# %%
decays = natural_decay_ops_D1() + quenching_ops("D1", gamma=QUENCHING_RATE) + wall_coll("D1")
freqs = np.linspace(-6, 6, 11)
#%%
last_ss = 0
rho_ss_list = []
for freq in freqs:
    if last_ss==0:
        ss = steadystate(H_atom_field_D1(-1, E_0_plus(0.01)) + H_atom(freq*2e9*pi, "D1"), c_op_list=decays)
    else:
        ss = steadystate(
                H_atom_field_D1(-1, E_0_plus(0.01)) + H_atom(freq*2e9*pi, "D1"), 
                c_op_list=decays, 
                method='iterative-gmres', 
                x0=operator_to_vector(last_ss)
            )
    last_ss = ss
    rho_ss_list.append(ss)
excited_pops = [sum(rho.diag()[8:]) for rho in rho_ss_list]
import pandas as pd
ser = pd.Series(excited_pops)
ser.index = freqs
delta = (freqs[-1] - freqs[0]) / len(freqs)
ser = ser.rolling(400, center=True, win_type='gaussian').mean(std=0.500/delta).dropna()
#%%
y = ser.dropna()
from lmfit.models import VoigtModel
mod = VoigtModel(prefix="p1_")+VoigtModel(prefix="p2_")+VoigtModel(prefix="p3_")+VoigtModel(prefix="p4_") #+ ConstantModel()
pars = mod.make_params()
for par in pars.keys():
    p = par.split("_")[1]
    if p=="amplitude":
        pars[par].set(value=1.6183e-08)
    elif p=="sigma":
        pars[par].set(value=0.05800174)
pars["p1_center"].set(value=-3.07192801)
pars["p2_center"].set(value=-2.21)
pars["p3_center"].set(value=3.77)
pars["p4_center"].set(value=4.6)
res = mod.fit(data=y.to_numpy(), params=pars, x=y.index.to_numpy())
res.plot();
plt.xlabel("Laser Detuning (GHz)")
res
# %%
import copy
pars = copy.copy(res.params)
for par in pars:
    p = par.split("_")[1]
    if p=="gamma":
        pars[par].set(vary=True)
res1 = mod.fit(data=y.to_numpy(), params=pars, x=y.index.to_numpy())
res1.plot();
plt.xlabel("Laser Detuning (GHz)")
res1

# %%
from lmfit.models import GaussianModel
mod = GaussianModel(prefix="p1_")+GaussianModel(prefix="p2_")+GaussianModel(prefix="p3_")+GaussianModel(prefix="p4_") #+ ConstantModel()
pars = mod.make_params()
for par in pars.keys():
    p = par.split("_")[1]
    if p=="amplitude":
        pars[par].set(value=1.6183e-08)
    elif p=="sigma":
        pars[par].set(value=0.5800174)
pars["p1_center"].set(value=-3.07192801)
pars["p2_center"].set(value=-2.21)
pars["p3_center"].set(value=3.77)
pars["p4_center"].set(value=4.6)
res = mod.fit(data=y.to_numpy(), params=pars, x=y.index.to_numpy())
res.plot();
plt.xlabel("Laser Detuning (GHz)")
res
# %%
