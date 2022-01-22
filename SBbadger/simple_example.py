
import generate
from scipy.stats import zipf


def in_dist(k):
    return k ** (-2)


def out_dist(k):
    return zipf.pmf(k, 3)


if __name__ == "__main__":

    generate.models(
        group_name="simple_example",
        n_species=100,
        kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
        in_dist=in_dist,
        out_dist=out_dist
        )

