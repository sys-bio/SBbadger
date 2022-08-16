
from SBbadger import generate_serial
from scipy.special import zeta


def in_dist(k):
    return k ** (-2) / zeta(2)


def out_dist(k):
    return k ** (-2) / zeta(2)


if __name__ == "__main__":

    generate_serial.models(

        group_name='hanekom2',
        n_models=1,
        n_species=10,
        out_dist=out_dist,
        in_dist=in_dist,
        rxn_prob=[.35, .30, .30, .05],
        kinetics=['hanekom', ['loguniform', 'uniform', 'loguniform'],
                                 ['v', 'k', 'keq'],
                                 [[0.01, 100], [0.0, 10], [0.01, 100]]],
        overwrite=True,
        rev_prob=.5,
        ic_params=['uniform', 0, 10],
        dist_plots=True,
        net_plots=True

    )
