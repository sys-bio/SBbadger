
from SBbadger import generate
import numpy as np


def bi_var_normal(x1, x2):
    return (1 / (2 * np.pi)) * np.exp(-(1 / 2) * ((((x1 - 10) / 1) ** 2) + (((x2 - 10) / 1) ** 2)))


# def joint_dist(k1, k2):
#     return k1**(-1.5) * k2**(-1.5) / (zeta(1.5) * zeta(1.5))


if __name__ == "__main__":

    generate.models(

        group_name='joint',
        n_models=1,
        n_species=100,
        kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
        overwrite=True,
        ic_params=['uniform', 0, 10],
        joint_dist=bi_var_normal,
        joint_range=[1, 19],
        dist_plots=True,
    )
