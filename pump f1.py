# %%
from init import *
# %%
h = H_atom(0, "D1")
# %%
t, v = h.eigenstates()
# %%
t
# %%
L = liouvillian(laser_sigma_plus_F1_FP_D1(1e-3), c_ops=dephasing_excited_states("D1") +
                natural_decay_ops_D1()+quenching_ops() + intra_F_ground_decay("D1") + F2_to_F1_ground_state_decay("D1"))

# %%
