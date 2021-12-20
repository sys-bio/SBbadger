
import processRNG
import os

# This example requires an existing set of distributions. Please see distributions_example.py.

processRNG.generate_networks(

    directory='models',
    group_name='test_group',
    overwrite=False,
    kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
    ic_params='trivial'
)
