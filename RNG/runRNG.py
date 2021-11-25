
from math import exp

from RNG.processRNG import runRNG

runRNG(
       group_name='test_group',
       overwrite=True,
       n_models=10,
       n_species=10,
       # kinetics=['hanekom', 'trivial', ['V', 'khs', 'keq']],
       kinetics=['hanekom', 'loguniform', ['V', 'khs', 'keq'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
       # kinetics=['hanekom', 'uniform', ['V', 'khs', 'keq'], [[0.0, 100], [0.0, 100], [0.0, 100]]],
       # kinetics=['hanekom', 'lognormal', ['V', 'khs', 'keq'], [[exp(1), 1], [exp(1), 1], [exp(1), 1]]],
       # kinetics=['hanekom', 'normal', ['V', 'khs', 'keq'], [[1, 1], [1, 1], [1, 1]]],

       # kinetics=['hanekom', 'loguniform', ['V', 'ks', 'kp', 'keq'], [[0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]],
       # kinetics=['hanekom', 'uniform', ['V', 'ks', 'kp', 'keq'], [[0.0, 100], [0.0, 100], [0.0, 100], [0.0, 100]]],
       # kinetics=['hanekom', 'lognormal', ['V', 'ks', 'kp', 'keq'], [[exp(1), 1], [exp(1), 1], [exp(1), 1], [exp(1), 1]]],
       # kinetics=['hanekom', 'normal', ['V', 'ks', 'kp', 'keq'], [[1, 1], [1, 1], [1, 1], [1, 1]]],

       rxn_prob=[0.25, 0.25, 0.25, 0.25],
       ICparams=[0, 10],
       # ICparams='trivial'
       )
