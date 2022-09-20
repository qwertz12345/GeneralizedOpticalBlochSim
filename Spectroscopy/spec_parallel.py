from numpy import linspace
from qutip import *

from init import (get_equally_ground_state_D1, laser_sigma_plus_F2_FP1_D1,
                  natural_decay_ops_D1, projector_excited_D1, quenching_ops)


def spec_solve1(freq):
    steps = 5000
    res_spec = mesolve(laser_sigma_plus_F2_FP1_D1(1e-3, det=freq),
                       rho0=get_equally_ground_state_D1(),
                       tlist=linspace(0, 1e-6, 100),
                       c_ops=natural_decay_ops_D1()+quenching_ops("D1"),
                       e_ops=[projector_excited_D1()],
                       options=Options(nsteps=steps))
    return res_spec.expect[0][-1]


