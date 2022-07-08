
from SBbadger import generate


if __name__ == "__main__":

    generate.models(

        group_name='mass_action3',
        kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],

    )
