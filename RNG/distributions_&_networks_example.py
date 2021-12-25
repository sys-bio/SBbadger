
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

    processRNG.generate_dists_networks(

        group_name='test_group',
        rxn_prob=[0.0, 0.0, 0.0, 1.0],
        n_models=10,
        n_species=10,
        out_dist=out_dist,
        kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
        overwrite=True,
        ic_params='trivial',
        plots=True,
        edge_type='metabolic',
        reaction_type='metabolic'
        # n_cpus=2
    )
