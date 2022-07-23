
from SBbadger import generate


if __name__ == "__main__":

    generate.rate_laws(
        kinetics=['modular_CM', ['loguniform', 'loguniform', 'loguniform', 'loguniform', 'loguniform',
                                 'loguniform', 'loguniform', 'loguniform', 'loguniform'],
                  ['ro', 'kf', 'kr', 'km', 'm',
                   'kms', 'ms', 'kma', 'ma'],
                  [[0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100],
                   [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]],
    )

