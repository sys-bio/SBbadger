
import processRNG
from scipy.special import zeta
import numpy as np


def in_dist(k):
    return k**(-2) / zeta(2)


def out_dist(k):
    return k**(-2) / zeta(2)


def bi_var_normal(x1, x2):
    return (1 / (2 * np.pi)) * np.exp(-(1 / 2) * ((((x1 - 10) / 1) ** 2) + (((x2 - 10) / 1) ** 2)))


processRNG.generate_distributions(
    group_name='test_group',
    overwrite=False,
    n_models=10,
    n_species=10,
    out_dist=out_dist,
    plots=True
)
