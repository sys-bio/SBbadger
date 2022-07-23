
from SBbadger import generate


def in_dist(k):
    return k ** (-2)


if __name__ == "__main__":

    generate.distributions(
        group_name="test",
        n_models=1,
        n_species=50,
        in_dist=in_dist,
        min_freq=1.0
        )
