from qutip import Options, mesolve
from numpy import linspace


def run_simulation(hamil_laser_ss):
    h, laser_ss, decays = hamil_laser_ss[0], hamil_laser_ss[1], hamil_laser_ss[2]
    time_evo_options = Options(nsteps=2**8 * 1000)
    res = mesolve(
        h,
        rho0=laser_ss,
        tlist=linspace(0, 1e-9, 5),
        c_ops=decays,
        options=time_evo_options,
    )
    return res
