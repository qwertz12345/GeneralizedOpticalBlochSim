# from qutip import steadystate
from genobs.lib import *

def calculate_steady_state(vars):
    laser_intens = vars[0]
    freq = vars[1]
    decays = vars[2]
    return steadystate(
        H_atom_field_D1(-1, E_0_plus(laser_intens*10)) + H_atom(freq*2e9*pi, "D1"), 
        c_op_list=decays
    )


if __name__=="__main__":
    print("test2")
    # from genobs.lib import *
    start_freq = -10
    stop_freq = 10
    points = 2000
    freqs = np.linspace(start_freq, stop_freq, points)    # GHz
    delta = (stop_freq - start_freq) / points

    laser_intens = 0.01    # mW/cm²
    gamma_quench = QUENCHING_RATE
    gamma_ground = 5e3

    doppler_broadening = 0.5    # GHz, FWHM
    doppler_sigma = doppler_broadening / (2*(2*np.log(2))**0.5)     # gaussian prop to exp(-1/2 * ((x-µ)/sigma)^2)
    doppler_sigma_points = doppler_sigma / delta

    decays = natural_decay_ops_D1() + quenching_ops("D1", gamma=gamma_quench) + wall_coll("D1", gamma=gamma_ground)
    # liouvillian_ops = [
    #     liouvillian(H=H_atom_field_D1(-1, E_0_plus(laser_intens*10)) 
    #                 + H_atom(freq*2e9*pi, "D1"), 
    #                 c_ops=decays) 
    #     for freq in freqs
    # ]
    # rho_ss_parallel = parallel_map(calculate_steady_state, liouvillian_ops)
    variables = [(laser_intens, f, decays) for f in freqs]
    rho_ss_parallel = parallel_map(calculate_steady_state, variables)

    excited_pops = [sum(rho.diag()[8:]) for rho in rho_ss_parallel]

    import pandas as pd
    ser = pd.Series(excited_pops)
    ser.index = freqs
    ser.plot()
    # plt.title("no doppler")
    # plt.xlabel("Laser Detuning (GHz)")

    # plt.figure()
    ser = ser.rolling(200, center=True, win_type='gaussian', min_periods=50).mean(std=doppler_sigma_points).dropna()
    ser.plot()
    plt.xlabel("Laser Detuning (GHz)")
    plt.legend(["no doppler", f"doppler broadening: {doppler_broadening:.2f}"])

    y = ser.dropna()
    from lmfit.models import VoigtModel
    mod = VoigtModel(prefix="p1_")+VoigtModel(prefix="p2_")+VoigtModel(prefix="p3_")+VoigtModel(prefix="p4_") #+ ConstantModel()
    pars = mod.make_params()
    for par in pars.keys():
        p = par.split("_")[1]
        if p=="amplitude":
            pars[par].set(value=1.5878e-08)
        elif p=="sigma":
            pars[par].set(value=0.21256648)
        # elif p=="gamma":
        #     pars[par].set(vary=True, value=0.15)
    pars["p1_center"].set(value=-3.07192801)
    pars["p2_center"].set(value=-2.21)
    pars["p3_center"].set(value=3.77)
    pars["p4_center"].set(value=4.6)
    res = mod.fit(data=y.to_numpy(), params=pars, x=y.index.to_numpy())
    import copy
    pars = copy.copy(res.params)
    for par in pars:
        p = par.split("_")[1]
        if p=="gamma":
            pars[par].set(vary=True, min=0, max=1)
    res = mod.fit(data=y.to_numpy(), params=pars, x=y.index.to_numpy())
    res.plot()
    plt.xlabel("Laser Detuning (GHz)")
    print(res.fit_report())
    plt.tight_layout()
    plt.show()
