
from RNG.processRNG import runRNG

runRNG(
       group_name='test_group',
       start_over=True,
       n_models=10,
       n_species=10,
       kinetics='mass_action',
       rev_prob=[0.5, 0.5, 0.5, 0.5]
       )
