# %%
from init import *

# %%

# 2nd Laser


def H_af_2nd_laser(power, pol=0):
    line = "D2"
    intens = get_probe_intensity(power)
    e0 = E_0_plus(intens)
    rab = rabi_D2_vector_component(E_field_component=e0)
    left = 1 / 2 * np.conjugate(rab) * sigma_q(q=pol, line=line)
    right = left.dag()
    return [[left + right, 'cos(delta_p * t)'],
            [1j * (left - right), "sin(delta_p * t)"]]


#%%
freq1 = (4.271676631815181 - 0.0729112 + 0.03)  # ghz, blue detuned from F=2 -> Fp=2 by 30 MHz
hamil1 = (H_atom_field_D2(0, E_0_plus(get_probe_intensity(1.5e-3))) 
        + H_atom(freq1 * 1e9 * 2 * pi, "D2"))
h = [hamil1] + H_af_2nd_laser(1.5e-3)
rho_ss_1 = steadystate(h[0]+h[1][0], c_op_list=natural_decay_ops_D2())
maplot(rho_ss_1)
#%%
times = np.linspace(0, 1e-5, 1001)
res = mesolve(h,
              rho0=rho_ss_1,
              tlist=times,
              c_ops=natural_decay_ops_D2(),
              args={'delta_p': (-6.834682610904290) * 2e9 * pi},
              options=Options(nsteps=2**2 * 1000),
              progress_bar=True)
plot_excited_states_time(res)
plot_ground_states_time(res)
maplot(res.states[-1])
rho_ee_t = [sum(state.diag()[8:]) for state in res.states]
plt.figure()
plt.plot(times, rho_ee_t)
plt.ylabel(r"$\rho_{ee}$")
plt.tight_layout()
# %%
