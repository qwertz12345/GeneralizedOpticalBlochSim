# %%
from init import *
# %%

def load_data(path_folder):
    import glob
    filenames = glob.glob(path_folder + "result*")
    files = []
    indeces_files = []
    for filename in filenames:
        indeces_files.append(int(filename.split("_")[1]))
        files.append(qload(filename))
    return indeces_files, files


# 2nd Laser
def H_af_2nd_laser(intens, pol=0):
    line = "D1"
    e0 = E_0_plus(intens)
    rab = rabi_D2_vector_component(E_field_component=e0)
    left = 1 / 2 * np.conjugate(rab) * sigma_q(q=pol, line=line)
    right = left.dag()
    return [[left + right, 'cos(delta_p * t)'],
            [1j * (left - right), "sin(delta_p * t)"]]


#%%
laser_pol = 0
ham_0 = H_atom(0, "D1")
freq1 = ham_0[-1, -1] - ham_0[5, 5]
freq2 = ham_0[5, 5] - ham_0[0, 0]
intens = SATURATION_INTENSITY_D2_SIGMA_PM_CYCLING/10
hamil1 = (H_atom_field_D1(laser_pol, E_0_plus(intens))
        + H_atom(freq1, "D1"))
offset = hamil1[-1, -1]
for k in range(16):
    hamil1[k, k] -= offset
h = [hamil1] + H_af_2nd_laser(intens, pol=laser_pol)
rho_ss_1 = steadystate(h[0]+h[1][0], c_op_list=natural_decay_ops_D1())
detunings = np.linspace(-500e6*2*pi, 500e6*2*pi, 201)
times = np.linspace(0, 1e-6, 200)
indeces = list(range(len(detunings)))
#%%
def calc_time_evo(ind):
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print(ind)
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    steps = 2**8 * 1000
    try:
        res = mesolve(h,
                rho0=rho_ss_1,
                tlist=times,
                c_ops=natural_decay_ops_D1(),
                args={'delta_p': freq2+detunings[ind]},
                options=Options(nsteps=steps),
                #progress_bar=False,
                )
    except Exception:
        # print("Failed----------------------!!!!!!!!!!!")
        # print(f"failed at {ind} with {steps} steps")
        return -1
    qsave(res, f"result_{ind}")
    # rho_ee_t = [sum(state.diag()[8:]) for state in res.states]
    # # fig, ax = plt.subplots()
    # # ax.plot(times, rho_ee_t)
    # # ax.set_ylabel(r"$\rho_{ee}$")
    # # ax.set_title(f"detuning: {detunings[ind]:.2e}")
    # # fig.savefig(f"tim_{ind}.png")
    # # plt.close(fig=fig)
    return res


def plot_exc_state_pops(indeces_, results):
    excited_pop=[]
    for elem in results:
        if elem==-1:
            excited_pop.append(None)
        else:
            excited_pop.append(sum(elem.states[-1].diag()[8:]))
    plt.plot([detunings[n] for n in indeces_], excited_pop, "o")
    plt.xlabel("Detuning Laser 2 from 1 (Hz)")
    plt.ylabel(r"$\rho_{ee}$")
    plt.tight_layout()
    plt.savefig("exc_evo.png")
    plt.show()
    print(len(results))
#%%
if __name__ == '__main__':
    results = parfor(calc_time_evo, indeces)#, progress_bar=True)
    plot_exc_state_pops(indeces, results)
