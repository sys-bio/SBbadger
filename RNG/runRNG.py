
from math import exp

from RNG.processRNG import runRNG

runRNG(
       group_name='test_group',
       overwrite=True,
       n_models=1000,
       n_species=10,

       # kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
       # kinetics=['mass_action', 'uniform', ['kf', 'kr', 'kc'], [[0.0, 100], [0.0, 100], [0.0, 100]]],
       kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
       # kinetics=['mass_action', 'norm', ['kf', 'kr', 'kc'], [[1, 1], [1, 1], [1, 1]]],
       # kinetics=['mass_action', 'lognorm', ['kf', 'kr', 'kc'], [[exp(1), 1], [exp(1), 1], [exp(1), 1]]],

       # kinetics=['mass_action', 'uniform', ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1',
       #                                      'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3'],
       #           [[0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100],
       #            [0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100]]],

       # kinetics=['mass_action', 'loguniform', ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1',
       #                                         'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3'],
       #           [[0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100],
       #            [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]],

       # kinetics=['mass_action', 'norm', ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1',
       #                                   'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3'],
       #            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
       #             [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]],

       # kinetics=['mass_action', 'lognorm', ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1',
       #                                      'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3'],
       #            [[exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1],
       #             [exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1]]],

       # kinetics=['hanekom', 'trivial', ['V', 'khs', 'keq']],
       # kinetics=['hanekom', 'uniform', ['V', 'khs', 'keq'], [[0.0, 100], [0.0, 100], [0.0, 100]]],
       # kinetics=['hanekom', 'loguniform', ['V', 'khs', 'keq'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
       # kinetics=['hanekom', 'normal', ['V', 'khs', 'keq'], [[1, 1], [1, 1], [1, 1]]],
       # kinetics=['hanekom', 'lognormal', ['V', 'khs', 'keq'], [[exp(1), 1], [exp(1), 1], [exp(1), 1]]],

       # kinetics=['hanekom', 'trivial', ['V', 'ks', 'kp', 'keq']],
       # kinetics=['hanekom', 'uniform', ['V', 'ks', 'kp', 'keq'], [[0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100]]],
       # kinetics=['hanekom', 'loguniform', ['V', 'ks', 'kp', 'keq'], [[0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]],
       # kinetics=['hanekom', 'normal', ['V', 'ks', 'kp', 'keq'], [[1, 1], [1, 1], [1, 1], [1, 1]]],
       # kinetics=['hanekom', 'lognormal', ['V', 'ks', 'kp', 'keq'], [[exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1]]],

       rev_prob=[0.5, 0.5, 0.5, 0.5],  # currently only valid for mass-action kinetics. Will be ignored if hanekom is specified.

       # rxn_prob=[1.0, 0.0, 0.0, 0.0],
       # rxn_prob=[0.0, 1.0, 0.0, 0.0],
       # rxn_prob=[0.0, 0.0, 1.0, 0.0],
       # rxn_prob=[0.0, 0.0, 0.0, 1.0],
       rxn_prob=[0.25, 0.25, 0.25, 0.25],

       # ICparams='trivial'
       ICparams=['dist', 0, 100],
       # ICparams=['list', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
       )
