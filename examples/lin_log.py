
from SBbadger import generate_serial
from scipy.stats import zipf


def in_dist(k):
    return k ** (-2)


def out_dist(k):
    return zipf.pmf(k, 3)


if __name__ == "__main__":

    model = generate_serial.models(

        group_name='lin_log',
        n_models=1,
        n_species=10,
        out_dist=out_dist,
        in_dist=in_dist,
        rxn_prob=[.35, .30, .30, .05],
        kinetics=['lin_log', ['uniform', 'uniform'],
                             ['v', 'rc'],
                             [[0.0, 100], [0.0, 100]]],
        overwrite=True,
        rev_prob=.5,
        ic_params=['uniform', 0, 10],
        dist_plots=True,
        net_plots=True
    )
