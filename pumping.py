#%%
from init import *
import time

#%%
hamil = laser_sigma_plus_F2_FP2_D1(
    OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL / 100)
maplot(hamil, annot=True)
#%%
L = liouvillian(
    hamil,
    c_ops=(
        natural_decay_ops_D1() 
        # + F1_to_F2_ground_state_decay("D1") 
        # + F2_to_F1_ground_state_decay("D1")
        )
    )
rho_ss = steadystate(L)
# %%
maplot(rho_ss)
# %%
start = time.perf_counter()

times = np.linspace(0, 50 / abs(hamil[-4,3]), 2000)
res = mesolve(L,
              tlist=times,
              rho0=get_equally_ground_state_D1(),
              options=Options(nsteps=2**5 * 1000))
stop = time.perf_counter()
print(f"""Time Evo "Exact" took {stop-start:.3f} seconds ------""")

qsave(res, "pumping_sigma_plus_f2_fp2_radecay")
#%%
plot_excited_states_time(res)
plot_ground_states_time(res)
#%%
plot_total_ground_pop(res)

#%%
f, a = maplot(res.states[-1])
f.suptitle("Time Evo")
f1, a1 = maplot(rho_ss)
f1.suptitle("Steady State")
# %%

hamil_approx = hamil.copy()
tmp = hamil_approx.full()
tmp[:3, 3:] = 0
tmp[3:, :3] = 0
tmp[8:11, :8] = 0
tmp[:8, 8:11] = 0
hamil_approx = Qobj(tmp)
maplot(hamil_approx, annot=True)
#%%
Lap = liouvillian(
    hamil_approx,
    c_ops=(
        natural_decay_ops_D1() 
        # + F1_to_F2_ground_state_decay("D1") 
        # + F2_to_F1_ground_state_decay("D1")
        )
    )
rho_ss_approx = steadystate(Lap, method="eigen")
# %%
maplot(rho_ss_approx)
# %%
start = time.perf_counter()
times = np.linspace(0, 50 / abs(hamil[-4,3]), 2000)
resa = mesolve(Lap,
              tlist=times,
              rho0=get_equally_ground_state_D1(),
              options=Options(nsteps=2**5 * 1000))
stop = time.perf_counter()
print(f"Time Evo Approx took {stop-start:.3f} seconds ------")
qsave(resa, "pumping_sigma_plus_f2_fp2_radecay_approx")

#%%
plot_excited_states_time(resa)
plot_ground_states_time(resa)
#%%
f, a = maplot(resa.states[-1])
f.suptitle("Time Evo Approx")
f1, a1 = maplot(rho_ss_approx)
f1.suptitle("Steady State Approx")
#%%
plot_total_ground_pop(resa)
#%%
plot_bar_ground_pop(resa.states[-1])
# %%
