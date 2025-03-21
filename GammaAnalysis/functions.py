import pymedphys
import numpy as np
import time

def GammaCalc(axe_ref, dose_ref, axe_evl, dose_evl,
              dose_percent_threshold, distance_mm_threshold,
              lower_percent_dose_cutoff, local_gamma, max_gamma):
    """计算 3D Gamma 通过率"""
    gamma_options = {
        'dose_percent_threshold': dose_percent_threshold,
        'distance_mm_threshold': distance_mm_threshold,
        'lower_percent_dose_cutoff': lower_percent_dose_cutoff,
        'interp_fraction': 10,  # 固定为 10，确保精度
        'max_gamma': max_gamma,
        'random_subset': None,
        'skip_once_passed': True,
        'local_gamma': local_gamma,
    }

    time1 = time.time()
    gamma = pymedphys.gamma(
        axe_ref, dose_ref,
        axe_evl, dose_evl,
        **gamma_options)

    valid_gamma = gamma[~np.isnan(gamma)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100
    print(f"Pass Ratio: {round(pass_ratio, 1)}%")
    print(f"Time: {time.time() - time1:.2f} s")

    return pass_ratio