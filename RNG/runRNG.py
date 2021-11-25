
from math import exp

from RNG.processRNG import runRNG

from buildNetworks import Settings
Settings.addDegradationSteps = True


"""
todo fix warnings you get when you run this program: 

D:\RNG\RNG\processRNG.py:28: SyntaxWarning: "is not" with a literal. Did you mean "!="?
  if jointDist and (inDist is not 'random' or outDist is not 'random'):
D:\RNG\RNG\processRNG.py:28: SyntaxWarning: "is not" with a literal. Did you mean "!="?
  if jointDist and (inDist is not 'random' or outDist is not 'random'):
  
"""

# main code should be sequestered away behind a main block. This is particularly true
# for multiprocessing
if __name__ == "__main__":
       runRNG(
              group_name='LargeHanekomNetworks',
              overwrite=True,
              n_models=30,
              n_species=100,
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

