
import processRNG
import os


processRNG.generate_networks(

    directory='models',
    group_name='test_group',
    overwrite=False,
    kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
    ic_params='trivial',
)
