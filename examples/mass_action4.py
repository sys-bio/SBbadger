
from SBbadger import generate_serial
from scipy.special import zeta


def in_dist(k):
    return k ** (-2) / zeta(2)


def out_dist(k):
    return k ** (-2) / zeta(2)


if __name__ == "__main__":

    model = generate_serial.models(

        group_name='mass_action4',
        n_models=1,
        n_species=20,
        out_dist=out_dist,
        in_dist=in_dist,
        rxn_prob=[.35, .30, .30, .05],
        kinetics=['mass_action', ['loguniform', 'loguniform', 'loguniform',
                                  'loguniform', 'loguniform', 'loguniform',
                                  'loguniform', 'loguniform', 'loguniform',
                                  'loguniform', 'loguniform', 'loguniform'],
                                 ['kf0', 'kr0', 'kc0',
                                  'kf1', 'kr1', 'kc1',
                                  'kf2', 'kr2', 'kc2',
                                  'kf3', 'kr3', 'kc3'],
                                 [[0.01, 100], [0.01, 100], [0.01, 100],
                                  [0.01, 100], [0.01, 100], [0.01, 100],
                                  [0.01, 100], [0.01, 100], [0.01, 100],
                                  [0.01, 100], [0.01, 100], [0.01, 100]]],
        overwrite=True,
        rev_prob=.5,
        ic_params=['uniform', 0, 10],
        dist_plots=True,
        net_plots=True
    )
