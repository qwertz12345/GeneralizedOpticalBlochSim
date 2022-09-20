#%%
from init import *
#%%
b_longitudinal = 0.1
resonant_mw_freqs = [
    b_longitudinal * 0.7e6 * 2 * pi * k for k in range(-3, 3 + 1)
]
# mw_freq = resonant_mw_freqs[0]
# power_laser = 1
# intens_laser = get_pump_intensity(power_laser)
decays = (
    natural_decay_ops_D1()
    # + quenching_ops("D1")
    + F2_to_F1_ground_state_decay("D1", gamma=1e3) 
    + F1_to_F2_ground_state_decay("D1", gamma=1e3)
    # + dephasing_excited_states("D1")
)

intens_laser = 0.01 * OFF_RESONANT_SATURATION_INTENSITY_D1_PI_POL
# print(f"laser intensity: {intens_laser*1e3/1e4:.2f} mW/cm**2")
hamil = laser_sigma_plus_F2_FP2_D1(intens_laser)
# remove off_reso terms (F=1 ...)
tmp = hamil.copy().full()
tmp[:3, 3:] = 0
tmp[3:, :3] = 0
tmp[8:11, :8] = 0
tmp[:8, 8:11] = 0
hamil_approx = Qobj(tmp)
rho_ss_laser = steadystate(hamil_approx, c_op_list=decays)
magnetic_field_mw = (10e3 / MU_BOHR) / 2**0.5
rabi_mw_est = MU_BOHR * magnetic_field_mw


def get_steadystate(det):
    ham = (hamil_approx + H_mw(magnetic_field_mw,
                               magnetic_field_mw,
                               0,
                               det_mw=det,
                               b_static_z=b_longitudinal))
    return steadystate(ham, c_op_list=decays, method="eigen")


#%%
if __name__ == "__main__":
    mw_freqs = -np.linspace(resonant_mw_freqs[0] - 10e3 ,
                           resonant_mw_freqs[0] + 10e3 , 1001)
    results = parallel_map(
        get_steadystate,
        mw_freqs,
    )
    rho_gg = [sum(res.diag()[:8]) for res in results]
    plt.plot(mw_freqs / (2 * pi * 1), rho_gg, "o")
    plt.xlabel("Detuning (Hz)")
    plt.tight_layout()
    plt.show()
