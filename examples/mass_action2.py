
from SBbadger import generate


if __name__ == "__main__":

    generate.models(

        group_name='mass_action2',
        n_models=10,
        n_species=10,
        n_reactions=10,
        kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]],

    )
