#%%
from init import *
#%%
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
]  # shift energies so that |F=1, mF=0> corresponds to 0 energy
ens = [
    en if k < 3 else en - energ_shifted[5]  # rotating frame
    for k, en in enumerate(energ_shifted)
]
#%%
Qobj(H_B(bx=1, by=0, bz=0).transform(F_states_reordered).full()**2)/9.7e12
#%%
Qobj(H_B(bx=0, by=0, bz=1).transform(F_states_reordered).full()**2)/9.7e12
#%%
Qobj(H_B(bx=0, by=1, bz=0).transform(F_states_reordered).full()**2)/9.7e12
# %%
