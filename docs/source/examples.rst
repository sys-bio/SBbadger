
Examples
########

Note that all examples can be found in the examples directory of SBbadger.

Mass-action
-----------

The first example demonstrates the use of the serialized version of SBbadgers model generator. The options
used here are

.. code-block:: console

    group_name: directory for output model files
    n_models:   number of models to generate
    n_species:  number of species per model
    out_dist:   distribution function that characterizes the out-edge node distribution
    in_dist:    distribution function that characterizes the in-edge node distribution
    rxn_prob:   reaction probabilities for UNI-UNI, BI-UNI, UNI-BI, and BI-BI reactions respectively
    kinetics:   information regarding the kinetics and parameters
    overwrite:  if True (default) all information in the named directory will be overwritten
    rev_prob:   probability that a reaction will be reversible
    ic_params:  initial value distribution
    dist_plots: write distribution plots to the named directory (default=False)
    net_plots:  write network plots to the named directory (default=False)

Note that each parameter can be defined with its own distribution

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta

    def in_dist(k):
        return k ** (-2) / zeta(2)

    def out_dist(k):
        return k ** (-2) / zeta(2)

    generate_serial.models(

        group_name='mass_action',
        n_models=1,
        n_species=10,
        out_dist=out_dist,
        in_dist=in_dist,
        rxn_prob=[.35, .30, .30, .05],
        kinetics=['mass_action', ['loguniform', 'loguniform', 'loguniform', 'uniform'],
                                 ['kf', 'kr', 'kc', 'deg'],
                                 [[0.01, 100], [0.01, 100], [0.01, 100], [1, 10]]],
        overwrite=True,
        rev_prob=.5,
        ic_params=['uniform', 0, 10],
        dist_plots=True,
        net_plots=True

    )

In the second mass-action example we see that the parallel ``generate.models()`` method is within
``if __name__ == "__main__":``. This must be so on Windows systems. The degree distribution functions
are omitted here, the result being that the nodes are chosen randomly for each additional reaction. Also
omitted is the deg option. The deg option, if present, adds a degradation reaction for each species in the
model. An additional option is added, ``n_reactions``, which provides a minimum number of reactions per
model and is only valid when no distribution functions are provided. Also, a global parameter distribution
type is provided instead of a type for each parameter. The distribution ranges/parameters remain
individualized.

.. code-block:: console

    from SBbadger import generate

    if __name__ == "__main__":

        generate.models(

            group_name='mass_action2',
            n_models=10,
            n_species=10,
            n_reactions=10,
            kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'],
                                                   [[0.01, 100], [0.01, 100], [0.01, 100]]],

        )

In the third mass-action example the parameters are set to ``trivial`` which just means they are set to ``1``.
This is an option designed for model fitting purposes where the parameters will be optimized with other
software.

.. code-block:: console

    from SBbadger import generate

    if __name__ == "__main__":

        generate.models(

            group_name='mass_action3',
            kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],

        )

Note that the different formats for the kinetics option above extend to the other kinetic formalisms below.

In the fourth mass-action example the distributions for the parameters are defined separately for each paramter
and reaction type.

.. code-block:: console

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

Generalized mass-action
-----------------------

In the following generalized mass-action example the ``ko`` parameters are the kinetic orders of the reactants
and products while the ``kor`` parameters are the kinetic orders of the regulating species. The gma_reg option
governs the number of regulators and whether or not they are activators or inhibitors. The list is a probability
distribution of the number of regulators (up to 3), i.e. [0, 1, 2, 3]. Thus in the example there is a 50% chance
of zero regulators and a 50% chance of one. The second term is the probability that the regulator is an inhibitor
or activator: (0: all inhibitors, 1: all activators). The kinetic orders of activators are positive and those of
inhibitors are negative. In future versions of SBbadger the maximum number of regulators will not be fixed. Please
see https://www.tandfonline.com/doi/abs/10.5661/bger-25-1 for more information on gma.

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta

    def in_dist(k):
        return k ** (-2) / zeta(2)

    def out_dist(k):
        return k ** (-2) / zeta(2)

    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='gma',
            n_models=1,
            n_species=10,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['gma', ['loguniform', 'loguniform', 'loguniform', 'uniform', 'uniform'],
                             ['kf', 'kr', 'kc', 'ko', 'kor'],
                                     [[0.01, 100], [0.01, 100], [0.01, 100], [0, 1], [0, 1]]],
            gma_reg=[[0.5, 0.5, 0, 0], .5],
            overwrite=True,
            rev_prob=.5,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

Lin-log
-------

Note that in the following lin-log example the scipy zipf distribution has been used for
the out-degree distribution.

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.stats import zipf

    def in_dist(k):
        return k ** (-2)

    def out_dist(k):
        return zipf.pmf(k, 3)

    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='lin_log',
            n_models=1,
            n_species=10,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['lin_log', ['uniform', 'uniform'],
                                 ['v', 'rc'],
                                 [[0.0, 100], [0.0, 100]]],
            overwrite=True,
            rev_prob=.5,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

Hanekom (generalized Michaelis-Menten)
--------------------------------------

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta


    def in_dist(k):
        return k ** (-2) / zeta(2)


    def out_dist(k):
        return k ** (-2) / zeta(2)


    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='hanekom1',
            n_models=1,
            n_species=10,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['hanekom', ['loguniform', 'uniform', 'uniform', 'loguniform'],
                                     ['v', 'ks', 'kp', 'keq'],
                                     [[0.01, 100], [0.0, 10], [0.0, 10], [0.01, 100]]],
            overwrite=True,
            rev_prob=.5,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

Note that in the following Hanekom example the half-saturation parameters for substrate and
product (``ks`` and ``kp``) are combined into a single ``k`` parameter.

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta


    def in_dist(k):
        return k ** (-2) / zeta(2)


    def out_dist(k):
        return k ** (-2) / zeta(2)


    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='hanekom2',
            n_models=1,
            n_species=10,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['hanekom', ['loguniform', 'uniform', 'loguniform'],
                                     ['v', 'k', 'keq'],
                                     [[0.01, 100], [0.0, 10], [0.01, 100]]],
            overwrite=True,
            rev_prob=.5,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

Liebermeister  (modular)
------------------------

In the following modular rate law example the ``km``, ``kma``, and ``kms`` parameters are the constants
for the reactants and products, allosteric regulators, and specific regulators respectively. This is also
true for the molecularities ``m``, ``ma``, and ``ms``. Options for individualized parameters governing
product and reactants, as well as activators and inhibitors, will be implemented in futures versions of
SBbadger. The ``mod_reg`` option governs the regulators. Like the ``gma_reg`` option for generalized
mass-action kinetics the list in the first element is the probability of having 0, 1, 2, or 3 regulators,
and the second element determines the probability that the regulator is an activator or inhibitor:
(0: all inhibitors, 1: all activators). The third term is the probability that the regulator is allosteric
or specific. The ``ro`` parameter can be set to ``trivial`` but will otherwise default to unform on
[0, 1] regardless of its designation within the ``kinetics`` option.

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta

    def in_dist(k):
        return k ** (-2) / zeta(2)

    def out_dist(k):
        return k ** (-2) / zeta(2)

    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='modular_CM',
            n_models=1,
            n_species=10,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['modular_CM', ['uniform', 'loguniform', 'loguniform', 'loguniform', 'loguniform',
                                     'loguniform', 'loguniform', 'loguniform', 'loguniform'],
                                    ['ro', 'kf', 'kr', 'km', 'm',
                                     'kms', 'ms', 'kma', 'ma'],
                                    [[0, 1], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100],
                                     [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]],
            mod_reg=[[0, 1, 0, 0], .5, .5],
            overwrite=True,
            rev_prob=0,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

The parameters are the same for all versions of modular rate laws including:

.. code-block:: console

    modular_CM
    modular_DM
    modular_SM
    modular_PM
    modular_FM

Please see https://pubmed.ncbi.nlm.nih.gov/20385728/ and https://arxiv.org/abs/2202.13004 for more
information.

Saturable and Cooperative
-------------------------

In the following example of saturable and cooperative kinetics formalism the regulatory option
``sc_reg`` mirrors that of ``gma_reg`` above. Please see
https://onlinelibrary.wiley.com/doi/epdf/10.1002/bit.21316 for more information on saturable
and cooperative kinetics.

.. code-block:: console

    from SBbadger import generate_serial
    from scipy.special import zeta

    def in_dist(k):
        return k ** (-2) / zeta(2)

    def out_dist(k):
        return k ** (-2) / zeta(2)

    if __name__ == "__main__":

        model = generate_serial.models(

            group_name='saturating_cooperative',
            n_models=1,
            n_species=20,
            out_dist=out_dist,
            in_dist=in_dist,
            rxn_prob=[.35, .30, .30, .05],
            kinetics=['saturating_cooperative', ['loguniform', 'loguniform', 'uniform', 'uniform'],
                                                ['v', 'k', 'n', 'nr'],
                                                [[0.01, 100], [0.01, 100], [0, 1], [0, 1]]],
            sc_reg=[[0.5, 0.5, 0, 0], 0.5],
            overwrite=True,
            rev_prob=.5,
            ic_params=['uniform', 0, 10],
            dist_plots=True,
            net_plots=True

        )

Joint distribution
-------------------------

The following example highlights the use of a joint distribution, specifically a bivariate normal distribution.
Note that joint distributions must be largely symmetrical so that sampling always results in equal numbers of in-edges
and out-edges. If a joint range is provided it is also assumed to be symmetrical, i.e., it applies to both variables.

.. code-block:: console

    from SBbadger import generate
    import numpy as np


    def bi_var_normal(x1, x2):
        return (1 / (2 * np.pi)) * np.exp(-(1 / 2) * ((((x1 - 10) / 1) ** 2) + (((x2 - 10) / 1) ** 2)))


    # def joint_dist(k1, k2):
    #     return k1**(-1.5) * k2**(-1.5) / (zeta(1.5) * zeta(1.5))


    if __name__ == "__main__":

        generate.models(

            group_name='joint',
            n_models=1,
            n_species=100,
            kinetics=['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]],
            overwrite=True,
            ic_params=['uniform', 0, 10],
            joint_dist=bi_var_normal,
            joint_range=[1, 19],
            dist_plots=True,
        )