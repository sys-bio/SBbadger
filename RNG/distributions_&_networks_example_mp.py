
import os

import processRNG
import processRNGmp
from scipy.special import zeta
import numpy as np
from scipy.stats import zipf, pareto


def in_dist(k):
    return k**(-2) / zeta(2)


def out_dist(k):
    return zipf.pmf(k, 2)


def bi_var_normal(x1, x2):
    return (1 / (2 * np.pi)) * np.exp(-(1 / 2) * ((((x1 - 10) / 1) ** 2) + (((x2 - 10) / 1) ** 2)))


if __name__ == "__main__":

    processRNGmp.generate_dists_networks(

        group_name='test_group',
        n_models=1000,
        n_species=100,
        out_dist=out_dist,
        kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
        overwrite=True,
        ic_params='trivial',
        plots=True,
        # edge_type='generic'
    )