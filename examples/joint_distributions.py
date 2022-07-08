
from SBbadger import generate
import numpy as np


def bi_var_normal(x1, x2):
    return (1 / (2 * np.pi)) * np.exp(-(1 / 2) * ((((x1 - 10) / 1) ** 2) + (((x2 - 10) / 1) ** 2)))

if __name__ == "__main__":

    generate.distributions(
        group_name="joint",
        n_models=1,
        n_species=100,
        joint_dist=bi_var_normal,
        joint_range=[1, 19],
        dist_plots=True
        )
